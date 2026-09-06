//! Code cache for JIT-compiled functions
//!
//! Stores compiled native code indexed by (module_id, function_index), with
//! support for invalidation (when a function is recompiled or patched).

use crate::jit::backend::traits::ExecutableCode;
use crate::jit::runtime::trampoline::JitEntryFn;
use crate::vm::object::LayoutId;
use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Composite key for the code cache: (module_id, func_index)
///
/// module_id disambiguates functions across different modules that may share
/// the same function index. Assigned by the cache via an atomic counter.
type CacheKey = (u64, u32);

/// Dependency generations captured before a function is lowered.
#[derive(Clone, Default)]
pub struct LayoutGenerationSnapshot {
    generations: FxHashMap<LayoutDependency, (Arc<AtomicU64>, u64)>,
}

impl LayoutGenerationSnapshot {
    /// Return the captured generation for one dependency.
    pub fn generation(&self, dependency: LayoutDependency) -> Option<u64> {
        self.generations
            .get(&dependency)
            .map(|(_, generation)| *generation)
    }

    fn dependencies(&self) -> impl Iterator<Item = LayoutDependency> + '_ {
        self.generations.keys().copied()
    }

    fn is_current(&self) -> bool {
        self.generations.values().all(|(counter, generation)| {
            counter.load(Ordering::Acquire) == *generation
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum LayoutDependency {
    AnyLayout,
    Layout(LayoutId),
}

/// Entry in the code cache
pub struct CacheEntry {
    /// The compiled executable code
    pub code: ExecutableCode,
    /// Whether this entry has been invalidated (e.g., function was recompiled)
    pub invalidated: AtomicBool,
    /// Module checksum for profile/cache invalidation plumbing.
    pub module_checksum: [u8; 32],
    /// Layout dependencies for generated-code invalidation.
    pub layout_dependencies: FxHashSet<LayoutDependency>,
    /// Dependency generations embedded in this compiled function.
    pub layout_generations: LayoutGenerationSnapshot,
}

/// Thread-safe cache of JIT-compiled function code
pub struct CodeCache {
    /// (module_id, func_index) → compiled code
    entries: RwLock<FxHashMap<CacheKey, CacheEntry>>,
    /// Total size of all cached code
    total_code_size: AtomicUsize,
    /// Maximum allowed total code size
    max_size: usize,
    /// Counter for assigning unique module IDs
    next_module_id: AtomicUsize,
    /// Module checksum → module_id mapping (for interpreter lookups)
    module_ids: RwLock<FxHashMap<[u8; 32], u64>>,
    /// Reverse module_id → module checksum mapping.
    module_checksums: RwLock<FxHashMap<u64, [u8; 32]>>,
    /// Layout dependency → cached functions that should be invalidated when that
    /// layout changes.
    layout_dependents: RwLock<FxHashMap<LayoutDependency, FxHashSet<CacheKey>>>,
    /// Lock-free generation used by current conservative object dependencies.
    any_layout_generation: Arc<AtomicU64>,
    /// Monotonic assumption generation for specific layout dependencies.
    layout_generations: RwLock<FxHashMap<LayoutDependency, Arc<AtomicU64>>>,
    /// Serializes publication with generation changes. Native generation checks
    /// remain lock-free through the counters stored in snapshots.
    layout_publication: RwLock<()>,
}

impl CodeCache {
    /// Create a new code cache with a maximum size limit (in bytes)
    pub fn new(max_size: usize) -> Self {
        CodeCache {
            entries: RwLock::new(FxHashMap::default()),
            total_code_size: AtomicUsize::new(0),
            max_size,
            next_module_id: AtomicUsize::new(0),
            module_ids: RwLock::new(FxHashMap::default()),
            module_checksums: RwLock::new(FxHashMap::default()),
            layout_dependents: RwLock::new(FxHashMap::default()),
            any_layout_generation: Arc::new(AtomicU64::new(0)),
            layout_generations: RwLock::new(FxHashMap::default()),
            layout_publication: RwLock::new(()),
        }
    }

    /// Allocate a unique module ID for use as the first part of the cache key
    pub fn allocate_module_id(&self) -> u64 {
        self.next_module_id.fetch_add(1, Ordering::Relaxed) as u64
    }

    /// Register a module by checksum and return its assigned module_id.
    ///
    /// If the module was already registered, returns the existing module_id.
    pub fn register_module(&self, checksum: [u8; 32]) -> u64 {
        let ids = self.module_ids.read();
        if let Some(&id) = ids.get(&checksum) {
            return id;
        }
        drop(ids);

        let id = self.allocate_module_id();
        self.module_ids.write().insert(checksum, id);
        self.module_checksums.write().insert(id, checksum);
        id
    }

    /// Look up the module_id for a given module checksum.
    ///
    /// Returns None if the module has not been JIT-compiled.
    pub fn module_id(&self, checksum: &[u8; 32]) -> Option<u64> {
        self.module_ids.read().get(checksum).copied()
    }

    /// Insert compiled code for a function
    ///
    /// Returns false if the cache is full and the entry was not inserted.
    pub fn insert(
        &self,
        module_id: u64,
        func_index: u32,
        code: ExecutableCode,
    ) -> bool {
        self.insert_with_dependencies(module_id, func_index, code, std::iter::empty())
    }

    /// Insert compiled code and record layout dependencies for generated-code invalidation.
    pub fn insert_with_dependencies<I>(
        &self,
        module_id: u64,
        func_index: u32,
        code: ExecutableCode,
        dependencies: I,
    ) -> bool
    where
        I: IntoIterator<Item = LayoutDependency>,
    {
        let layout_generations = self.snapshot_layout_generations(dependencies);
        self.insert_with_layout_generations(module_id, func_index, code, layout_generations)
    }

    /// Insert code only if the dependency generations captured before lowering
    /// are still current. This prevents publishing code compiled across an
    /// invalidation.
    pub fn insert_with_layout_generations(
        &self,
        module_id: u64,
        func_index: u32,
        code: ExecutableCode,
        layout_generations: LayoutGenerationSnapshot,
    ) -> bool {
        let key = (module_id, func_index);
        let code_size = code.code_size;
        let current = self.total_code_size.load(Ordering::Relaxed);
        if current + code_size > self.max_size {
            return false;
        }
        let module_checksum = self
            .module_checksums
            .read()
            .get(&module_id)
            .copied()
            .unwrap_or([0; 32]);
        let layout_dependencies: FxHashSet<_> = layout_generations.dependencies().collect();

        // Invalidation takes the write side, increments generations, and marks
        // existing dependents before another publication can complete.
        let _publication = self.layout_publication.read();
        if !layout_generations.is_current() {
            return false;
        }
        let mut entries = self.entries.write();
        // Remove old entry size if replacing
        if let Some(old) = entries.remove(&key) {
            self.total_code_size
                .fetch_sub(old.code.code_size, Ordering::Relaxed);
            let mut dependents = self.layout_dependents.write();
            for dep in old.layout_dependencies {
                if let Some(keys) = dependents.get_mut(&dep) {
                    keys.remove(&key);
                    if keys.is_empty() {
                        dependents.remove(&dep);
                    }
                }
            }
        }

        self.total_code_size.fetch_add(code_size, Ordering::Relaxed);
        entries.insert(
            key,
            CacheEntry {
                code,
                invalidated: AtomicBool::new(false),
                module_checksum,
                layout_dependencies: layout_dependencies.clone(),
                layout_generations,
            },
        );
        if !layout_dependencies.is_empty() {
            let mut dependents = self.layout_dependents.write();
            for dep in layout_dependencies {
                dependents.entry(dep).or_default().insert(key);
            }
        }
        true
    }

    /// Look up the JIT entry function for a (module_id, func_index) pair
    ///
    /// Returns None if the function isn't compiled or has been invalidated.
    pub fn get(&self, module_id: u64, func_index: u32) -> Option<JitEntryFn> {
        let key = (module_id, func_index);
        let (code_ptr, entry_offset, invalidated, generations) = {
            let entries = self.entries.read();
            let entry = entries.get(&key)?;
            (
                entry.code.code_ptr,
                entry.code.entry_offset,
                entry.invalidated.load(Ordering::Acquire),
                entry.layout_generations.clone(),
            )
        };
        if invalidated || !self.layout_generations_match(&generations) {
            return None;
        }
        // Safety: entry_offset is within code bounds (verified at finalize time)
        let fn_ptr = unsafe { code_ptr.add(entry_offset) };
        Some(unsafe { std::mem::transmute::<*const u8, JitEntryFn>(fn_ptr) })
    }

    /// Invalidate a cached function (e.g., when deoptimizing)
    pub fn invalidate(&self, module_id: u64, func_index: u32) {
        let key = (module_id, func_index);
        let entries = self.entries.read();
        if let Some(entry) = entries.get(&key) {
            entry.invalidated.store(true, Ordering::Release);
        }
    }

    /// Capture the current generation for each dependency.
    pub fn snapshot_layout_generations<I>(&self, dependencies: I) -> LayoutGenerationSnapshot
    where
        I: IntoIterator<Item = LayoutDependency>,
    {
        let dependencies: FxHashSet<_> = dependencies.into_iter().collect();
        let generations = dependencies
            .into_iter()
            .map(|dependency| {
                let counter = self.layout_generation_counter(dependency);
                let generation = counter.load(Ordering::Acquire);
                (dependency, (counter, generation))
            })
            .collect();
        LayoutGenerationSnapshot { generations }
    }

    fn layout_generation_counter(&self, dependency: LayoutDependency) -> Arc<AtomicU64> {
        match dependency {
            LayoutDependency::AnyLayout => self.any_layout_generation.clone(),
            LayoutDependency::Layout(_) => self
                .layout_generations
                .write()
                .entry(dependency)
                .or_insert_with(|| Arc::new(AtomicU64::new(0)))
                .clone(),
        }
    }

    /// Return the current generation for one dependency.
    pub fn layout_generation(&self, dependency: LayoutDependency) -> u64 {
        self.layout_generation_counter(dependency)
            .load(Ordering::Acquire)
    }

    /// Check a generation embedded in native code at an assumption site.
    pub fn layout_generation_matches(
        &self,
        dependency: LayoutDependency,
        expected: u64,
    ) -> bool {
        self.layout_generation(dependency) == expected
    }

    fn layout_generations_match(&self, expected: &LayoutGenerationSnapshot) -> bool {
        expected.is_current()
    }

    /// Invalidate every compiled function that depends on the given layout.
    ///
    /// Returns `(module_checksum, func_index)` pairs so external profiling state
    /// can reset its `jit_available` flags too.
    pub fn invalidate_layout(&self, layout_id: LayoutId) -> Vec<([u8; 32], u32)> {
        let _publication = self.layout_publication.write();
        for dependency in [
            LayoutDependency::AnyLayout,
            LayoutDependency::Layout(layout_id),
        ] {
            self.layout_generation_counter(dependency)
                .fetch_add(1, Ordering::AcqRel);
        }

        let mut affected_keys: FxHashSet<CacheKey> = FxHashSet::default();
        {
            let dependents = self.layout_dependents.read();
            if let Some(keys) = dependents.get(&LayoutDependency::AnyLayout) {
                affected_keys.extend(keys.iter().copied());
            }
            if let Some(keys) = dependents.get(&LayoutDependency::Layout(layout_id)) {
                affected_keys.extend(keys.iter().copied());
            }
        }

        let entries = self.entries.read();
        affected_keys
            .into_iter()
            .filter_map(|key| {
                let entry = entries.get(&key)?;
                entry.invalidated.store(true, Ordering::Release);
                Some((entry.module_checksum, key.1))
            })
            .collect()
    }

    /// Check if a function has been compiled and is valid
    pub fn contains(&self, module_id: u64, func_index: u32) -> bool {
        let key = (module_id, func_index);
        let entry_state = self.entries.read().get(&key).map(|entry| {
            (
                entry.invalidated.load(Ordering::Acquire),
                entry.layout_generations.clone(),
            )
        });
        entry_state
            .map(|(invalidated, generations)| {
                !invalidated && self.layout_generations_match(&generations)
            })
            .unwrap_or(false)
    }

    /// Total size of cached code
    pub fn total_size(&self) -> usize {
        self.total_code_size.load(Ordering::Relaxed)
    }

    /// Number of cached functions (including invalidated)
    pub fn entry_count(&self) -> usize {
        self.entries.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dummy_code(size: usize) -> ExecutableCode {
        // Safety: this is test code — we never execute these pointers
        ExecutableCode {
            code_ptr: std::ptr::null(),
            code_size: size,
            entry_offset: 0,
            stack_maps: vec![],
            deopt_info: vec![],
        }
    }

    #[test]
    fn test_insert_and_contains() {
        let cache = CodeCache::new(1024);
        let mid = cache.allocate_module_id();
        assert!(!cache.contains(mid, 0));

        let inserted = cache.insert(mid, 0, make_dummy_code(100));
        assert!(inserted);
        assert!(cache.contains(mid, 0));
        assert_eq!(cache.total_size(), 100);
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn test_invalidate() {
        let cache = CodeCache::new(1024);
        let mid = cache.allocate_module_id();
        cache.insert(mid, 0, make_dummy_code(100));
        assert!(cache.contains(mid, 0));

        cache.invalidate(mid, 0);
        assert!(!cache.contains(mid, 0));
        // Entry still exists (just invalidated)
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn test_cache_full() {
        let cache = CodeCache::new(200);
        let mid = cache.allocate_module_id();
        assert!(cache.insert(mid, 0, make_dummy_code(100)));
        assert!(cache.insert(mid, 1, make_dummy_code(100)));
        // Cache is now full (200/200)
        assert!(!cache.insert(mid, 2, make_dummy_code(100)));
        assert_eq!(cache.entry_count(), 2);
    }

    #[test]
    fn test_replace_entry() {
        let cache = CodeCache::new(1024);
        let mid = cache.allocate_module_id();
        cache.insert(mid, 0, make_dummy_code(100));
        assert_eq!(cache.total_size(), 100);

        // Replace with larger code
        cache.insert(mid, 0, make_dummy_code(200));
        assert_eq!(cache.total_size(), 200);
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn test_different_modules_same_func_index() {
        let cache = CodeCache::new(1024);
        let mid1 = cache.allocate_module_id();
        let mid2 = cache.allocate_module_id();

        cache.insert(mid1, 0, make_dummy_code(100));
        cache.insert(mid2, 0, make_dummy_code(100));

        assert!(cache.contains(mid1, 0));
        assert!(cache.contains(mid2, 0));
        assert_eq!(cache.entry_count(), 2);

        // Invalidate only module 1's function
        cache.invalidate(mid1, 0);
        assert!(!cache.contains(mid1, 0));
        assert!(cache.contains(mid2, 0));
    }

    #[test]
    fn test_allocate_module_id_increments() {
        let cache = CodeCache::new(1024);
        let id1 = cache.allocate_module_id();
        let id2 = cache.allocate_module_id();
        let id3 = cache.allocate_module_id();
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
    }

    #[test]
    fn test_invalidate_layout_marks_matching_entries() {
        let cache = CodeCache::new(1024);
        let checksum = [7; 32];
        let mid = cache.register_module(checksum);
        assert!(cache.insert_with_dependencies(
            mid,
            3,
            make_dummy_code(64),
            [LayoutDependency::Layout(42)]
        ));
        assert!(cache.contains(mid, 3));

        let affected = cache.invalidate_layout(42);
        assert_eq!(affected, vec![(checksum, 3)]);
        assert!(!cache.contains(mid, 3));
    }

    #[test]
    fn layout_invalidation_advances_only_matching_generations() {
        let cache = CodeCache::new(1024);
        let any = cache.layout_generation(LayoutDependency::AnyLayout);
        let matching = cache.layout_generation(LayoutDependency::Layout(42));
        let other = cache.layout_generation(LayoutDependency::Layout(7));

        cache.invalidate_layout(42);

        assert_eq!(cache.layout_generation(LayoutDependency::AnyLayout), any + 1);
        assert_eq!(
            cache.layout_generation(LayoutDependency::Layout(42)),
            matching + 1
        );
        assert_eq!(cache.layout_generation(LayoutDependency::Layout(7)), other);
    }

    #[test]
    fn stale_compilation_snapshot_is_not_inserted() {
        let cache = CodeCache::new(1024);
        let mid = cache.register_module([3; 32]);
        let snapshot = cache.snapshot_layout_generations([LayoutDependency::AnyLayout]);

        cache.invalidate_layout(42);

        assert!(!cache.insert_with_layout_generations(
            mid,
            0,
            make_dummy_code(64),
            snapshot,
        ));
        assert_eq!(cache.entry_count(), 0);
    }

    #[test]
    fn invalidating_one_layout_preserves_other_specific_dependencies() {
        let cache = CodeCache::new(1024);
        let first = cache.register_module([4; 32]);
        let second = cache.register_module([5; 32]);
        let any = cache.register_module([6; 32]);
        assert!(cache.insert_with_dependencies(
            first,
            0,
            make_dummy_code(64),
            [LayoutDependency::Layout(42)],
        ));
        assert!(cache.insert_with_dependencies(
            second,
            0,
            make_dummy_code(64),
            [LayoutDependency::Layout(7)],
        ));
        assert!(cache.insert_with_dependencies(
            any,
            0,
            make_dummy_code(64),
            [LayoutDependency::AnyLayout],
        ));

        cache.invalidate_layout(42);

        assert!(!cache.contains(first, 0));
        assert!(cache.contains(second, 0));
        assert!(!cache.contains(any, 0));
    }

    #[test]
    fn test_any_layout_dependency_invalidates_on_any_layout_change() {
        let cache = CodeCache::new(1024);
        let checksum = [9; 32];
        let mid = cache.register_module(checksum);
        assert!(cache.insert_with_dependencies(
            mid,
            1,
            make_dummy_code(64),
            [LayoutDependency::AnyLayout]
        ));
        let affected = cache.invalidate_layout(999);
        assert_eq!(affected, vec![(checksum, 1)]);
    }
}
