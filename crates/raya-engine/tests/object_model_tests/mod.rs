//! Integration tests for Object Model (Milestone 1.6)
//!
//! Tests cover:
//! - Object creation and field access
//! - Array operations (creation, access, bounds checking)
//! - String operations (concatenation, length)
//! - Method dispatch via vtables
//! - GC integration with objects

use raya_engine::compiler::{ClassDef, Function, Module, Opcode};
use raya_engine::vm::interpreter::Vm;
use raya_engine::vm::object::layout_id_from_ordered_names;
use raya_engine::vm::value::Value;
use std::sync::Arc;

fn class_def(name: &str, field_count: usize, parent_id: Option<u32>) -> ClassDef {
    ClassDef {
        name: name.to_string(),
        field_count,
        parent_id,
        methods: Vec::new(),
    }
}

#[test]
fn test_object_creation_and_field_access() {
    let mut vm = Vm::new();

    // Bytecode: new Point(), set x=10, y=20, read x
    let mut module = Module::new("test".to_string());
    module.classes.push(class_def("Point", 2, None));
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 1,
        code: vec![
            // new Point() -> local 0
            Opcode::NewType as u8,
            0,
            0, // class index 0
            Opcode::StoreLocal as u8,
            0,
            0,
            // obj.x = 10
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ConstI32 as u8,
            10,
            0,
            0,
            0,
            Opcode::StoreFieldExact as u8,
            0,
            0, // field offset 0
            // obj.y = 20
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ConstI32 as u8,
            20,
            0,
            0,
            0,
            Opcode::StoreFieldExact as u8,
            1,
            0, // field offset 1
            // return obj.x
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::LoadFieldExact as u8,
            0,
            0, // field offset 0
            Opcode::Return as u8,
        ],
    };
    module.functions.push(main_fn);

    let result = vm.execute(&module).unwrap();
    assert_eq!(result, Value::i32(10));
}

#[test]
fn test_array_creation_and_access() {
    // Bytecode: arr = new Array(3), arr[0]=10, arr[1]=20, arr[2]=30, return arr[1]
    let mut module = Module::new("test".to_string());
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 1,
        code: vec![
            // new Array(3) -> local 0
            Opcode::ConstI32 as u8,
            3,
            0,
            0,
            0, // length
            Opcode::NewArray as u8,
            0,
            0, // type index 0
            0,
            0,
            Opcode::StoreLocal as u8,
            0,
            0,
            // arr[0] = 10
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ConstI32 as u8,
            0,
            0,
            0,
            0, // index 0
            Opcode::ConstI32 as u8,
            10,
            0,
            0,
            0, // value 10
            Opcode::StoreElem as u8,
            // arr[1] = 20
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ConstI32 as u8,
            1,
            0,
            0,
            0, // index 1
            Opcode::ConstI32 as u8,
            20,
            0,
            0,
            0, // value 20
            Opcode::StoreElem as u8,
            // arr[2] = 30
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ConstI32 as u8,
            2,
            0,
            0,
            0, // index 2
            Opcode::ConstI32 as u8,
            30,
            0,
            0,
            0, // value 30
            Opcode::StoreElem as u8,
            // return arr[1]
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ConstI32 as u8,
            1,
            0,
            0,
            0, // index 1
            Opcode::LoadElem as u8,
            Opcode::Return as u8,
        ],
    };
    module.functions.push(main_fn);

    let mut vm = Vm::new();
    let result = vm.execute(&module).unwrap();
    assert_eq!(result, Value::i32(20));
}

#[test]
fn test_array_length() {
    // Bytecode: arr = new Array(5), return arr.length
    let mut module = Module::new("test".to_string());
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 1,
        code: vec![
            // new Array(5) -> local 0
            Opcode::ConstI32 as u8,
            5,
            0,
            0,
            0, // length
            Opcode::NewArray as u8,
            0,
            0, // type index 0
            0,
            0,
            Opcode::StoreLocal as u8,
            0,
            0,
            // return arr.length
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ArrayLen as u8,
            Opcode::Return as u8,
        ],
    };
    module.functions.push(main_fn);

    let mut vm = Vm::new();
    let result = vm.execute(&module).unwrap();
    assert_eq!(result, Value::i32(5));
}

#[test]
fn test_multiple_objects() {
    let mut vm = Vm::new();

    // Bytecode: create Point with x=5, y=10, create Rectangle with x1=0, y1=0, x2=100, y2=50
    // return Point.x + Point.y + Rectangle.x2
    let mut module = Module::new("test".to_string());
    module.classes.push(class_def("Point", 2, None));
    module.classes.push(class_def("Rectangle", 4, None));
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 2,
        code: vec![
            // Point -> local 0
            Opcode::NewType as u8,
            0,
            0, // class 0
            Opcode::StoreLocal as u8,
            0,
            0,
            // Point.x = 5
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ConstI32 as u8,
            5,
            0,
            0,
            0,
            Opcode::StoreFieldExact as u8,
            0,
            0,
            // Point.y = 10
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ConstI32 as u8,
            10,
            0,
            0,
            0,
            Opcode::StoreFieldExact as u8,
            1,
            0,
            // Rectangle -> local 1
            Opcode::NewType as u8,
            1,
            0, // class 1
            Opcode::StoreLocal as u8,
            1,
            0,
            // Rectangle.x2 = 100 (field index 2)
            Opcode::LoadLocal as u8,
            1,
            0,
            Opcode::ConstI32 as u8,
            100,
            0,
            0,
            0,
            Opcode::StoreFieldExact as u8,
            2,
            0,
            // Calculate Point.x + Point.y + Rectangle.x2
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::LoadFieldExact as u8,
            0,
            0, // Point.x
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::LoadFieldExact as u8,
            1,
            0, // Point.y
            Opcode::Iadd as u8,
            Opcode::LoadLocal as u8,
            1,
            0,
            Opcode::LoadFieldExact as u8,
            2,
            0, // Rectangle.x2
            Opcode::Iadd as u8,
            Opcode::Return as u8,
        ],
    };
    module.functions.push(main_fn);

    let result = vm.execute(&module).unwrap();
    assert_eq!(result, Value::i32(115)); // 5 + 10 + 100
}

#[test]
fn test_object_with_gc() {
    // Test that objects survive GC when they're referenced
    let mut vm = Vm::new();

    // Create an object, store it in a local, trigger GC, access it
    let mut module = Module::new("test".to_string());
    module.classes.push(class_def("Point", 2, None));
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 1,
        code: vec![
            // Create Point
            Opcode::NewType as u8,
            0,
            0,
            Opcode::StoreLocal as u8,
            0,
            0,
            // Set x = 42
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ConstI32 as u8,
            42,
            0,
            0,
            0,
            Opcode::StoreFieldExact as u8,
            0,
            0,
            // Load x and return it (object should survive GC)
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::LoadFieldExact as u8,
            0,
            0,
            Opcode::Return as u8,
        ],
    };
    module.functions.push(main_fn);

    let result = vm.execute(&module).unwrap();
    assert_eq!(result, Value::i32(42));

    // Trigger GC after execution - object is no longer reachable
    vm.collect_garbage();
}

#[test]
fn test_object_literal() {
    // Test OBJECT_LITERAL + INIT_OBJECT opcodes
    // Creates Point{x: 10, y: 20} using literal syntax
    let mut vm = Vm::new();
    let layout_id = layout_id_from_ordered_names(&["x".to_string(), "y".to_string()]);
    let mut module = Module::new("test".to_string());
    let mut code = vec![Opcode::ObjectLiteral as u8];
    code.extend_from_slice(&layout_id.to_le_bytes());
    code.extend_from_slice(&2u16.to_le_bytes());
    code.extend_from_slice(&[Opcode::ConstI32 as u8, 10, 0, 0, 0]);
    code.extend_from_slice(&[Opcode::InitObject as u8, 0, 0]);
    code.extend_from_slice(&[Opcode::ConstI32 as u8, 20, 0, 0, 0]);
    code.extend_from_slice(&[Opcode::InitObject as u8, 1, 0]);
    code.extend_from_slice(&[Opcode::LoadFieldExact as u8, 0, 0, Opcode::Return as u8]);
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 0,
        code,
    };
    module.functions.push(main_fn);

    let result = vm.execute(&module).unwrap();
    assert_eq!(result, Value::i32(10));
}

#[test]
fn test_array_literal() {
    // Test ARRAY_LITERAL opcode
    // ARRAY_LITERAL pops elements from stack and creates array
    // So we push elements first: [10, 20, 30]
    let mut module = Module::new("test".to_string());
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 0,
        code: vec![
            // Push elements in order (first element first)
            Opcode::ConstI32 as u8,
            10,
            0,
            0,
            0,
            Opcode::ConstI32 as u8,
            20,
            0,
            0,
            0,
            Opcode::ConstI32 as u8,
            30,
            0,
            0,
            0,
            // ARRAY_LITERAL type=0, length=3
            // Pops 3 elements, creates array [10, 20, 30]
            Opcode::ArrayLiteral as u8,
            0,
            0,
            0,
            0, // type index 0 (u32)
            3,
            0,
            0,
            0, // length 3 (u32)
            // Array is now on stack with all elements set
            // Read element 1 to verify (should be 20)
            Opcode::ConstI32 as u8,
            1,
            0,
            0,
            0,
            Opcode::LoadElem as u8,
            Opcode::Return as u8,
        ],
    };
    module.functions.push(main_fn);

    let mut vm = Vm::new();
    let result = vm.execute(&module).unwrap();
    assert_eq!(result, Value::i32(20));
}

#[test]
fn test_static_fields() {
    // Test LOAD_STATIC + STORE_STATIC opcodes
    let mut vm = Vm::new();

    let mut module = Module::new("test".to_string());
    module.classes.push(class_def("Counter", 2, None));
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 0,
        code: vec![
            // Load static field 1 (initial value 100)
            Opcode::LoadStatic as u8,
            0,
            0, // class index 0
            1,
            0, // field offset 1
            // Add 42
            Opcode::ConstI32 as u8,
            42,
            0,
            0,
            0,
            Opcode::Iadd as u8,
            // Store back to static field 0
            Opcode::StoreStatic as u8,
            0,
            0, // class index 0
            0,
            0, // field offset 0
            // Load static field 0 to verify
            Opcode::LoadStatic as u8,
            0,
            0, // class index 0
            0,
            0, // field offset 0
            Opcode::Return as u8,
        ],
    };
    module.functions.push(main_fn);

    let runtime_module = Arc::new(Module::decode(&module.encode()).unwrap());
    vm.shared_state().register_module(runtime_module.clone()).unwrap();
    let nominal_type_base = vm
        .shared_state()
        .module_layouts
        .read()
        .get(&runtime_module.checksum)
        .unwrap()
        .nominal_type_base;
    vm.shared_state()
        .classes
        .write()
        .get_class_mut(nominal_type_base)
        .unwrap()
        .static_fields = vec![Value::i32(0), Value::i32(100)];

    let result = vm.execute(runtime_module.as_ref()).unwrap();
    assert_eq!(result, Value::i32(142)); // 100 + 42
}

#[test]
fn test_optional_field_non_null() {
    // Test OPTIONAL_FIELD opcode with non-null object
    let mut vm = Vm::new();
    let mut module = Module::new("test".to_string());
    module.classes.push(class_def("Point", 2, None));
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 1,
        code: vec![
            // Create Point
            Opcode::NewType as u8,
            0,
            0,
            Opcode::StoreLocal as u8,
            0,
            0,
            // Set x = 42
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ConstI32 as u8,
            42,
            0,
            0,
            0,
            Opcode::StoreFieldExact as u8,
            0,
            0,
            // Load object and access optional field
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::OptionalFieldExact as u8,
            0,
            0, // field offset 0
            Opcode::Return as u8,
        ],
    };
    module.functions.push(main_fn);

    let result = vm.execute(&module).unwrap();
    assert_eq!(result, Value::i32(42));
}

#[test]
fn test_optional_field_null() {
    // Test OPTIONAL_FIELD opcode with null object
    let mut vm = Vm::new();
    let mut module = Module::new("test".to_string());
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 0,
        code: vec![
            // Push null
            Opcode::ConstNull as u8,
            // Access optional field (should return null)
            Opcode::OptionalFieldExact as u8,
            0,
            0, // field offset 0
            Opcode::Return as u8,
        ],
    };
    module.functions.push(main_fn);

    let result = vm.execute(&module).unwrap();
    assert_eq!(result, Value::null());
}

#[test]
fn test_constructor_no_args() {
    // Test CALL_CONSTRUCTOR with no arguments
    let mut vm = Vm::new();

    let mut module = Module::new("test".to_string());
    module.classes.push(class_def("Point", 2, None));

    // Main function: calls constructor with no args
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 1,
        code: vec![
            // CALL_CONSTRUCTOR class=0, arg_count=0
            Opcode::CallConstructor as u8,
            0,
            0,
            0,
            0, // class index 0
            0,
            0, // arg count 0
            // Store returned object
            Opcode::StoreLocal as u8,
            0,
            0,
            // Load object and set field directly
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ConstI32 as u8,
            42,
            0,
            0,
            0,
            Opcode::StoreFieldExact as u8,
            0,
            0,
            // Load and return field 0 to verify
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::LoadFieldExact as u8,
            0,
            0,
            Opcode::Return as u8,
        ],
    };
    module.functions.push(main_fn);

    // Empty constructor
    let constructor_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "Point::constructor".to_string(),
        param_count: 1, // just this
        local_count: 1, // total locals = 1 (this only)
        code: vec![
            // Just return null
            Opcode::ConstNull as u8,
            Opcode::Return as u8,
        ],
    };
    module.functions.push(constructor_fn);

    let result = vm.execute(&module).unwrap();
    assert_eq!(result, Value::i32(42));
}

#[test]
fn test_constructor_basic() {
    // Test CALL_CONSTRUCTOR opcode
    let mut vm = Vm::new();

    let mut module = Module::new("test".to_string());
    module.classes.push(class_def("Point", 2, None));

    // Main function: calls constructor with args 10, 20
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 1,
        code: vec![
            // Push constructor arguments
            Opcode::ConstI32 as u8,
            10,
            0,
            0,
            0, // arg 0 (x)
            Opcode::ConstI32 as u8,
            20,
            0,
            0,
            0, // arg 1 (y)
            // CALL_CONSTRUCTOR class=0, arg_count=2
            Opcode::CallConstructor as u8,
            0,
            0,
            0,
            0, // class index 0
            2,
            0, // arg count 2
            // Store returned object
            Opcode::StoreLocal as u8,
            0,
            0,
            // Load and return field 0 to verify
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::LoadFieldExact as u8,
            0,
            0,
            Opcode::Return as u8,
        ],
    };
    module.functions.push(main_fn);

    // Constructor function: initializes fields from args
    let constructor_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "Point::constructor".to_string(),
        param_count: 3, // this + 2 args
        local_count: 3, // total locals = 3 (this + x + y)
        code: vec![
            // Load 'this' (param 0)
            Opcode::LoadLocal as u8,
            0,
            0,
            // Load x (param 1)
            Opcode::LoadLocal as u8,
            1,
            0,
            // Set this.x = x
            Opcode::StoreFieldExact as u8,
            0,
            0,
            // Load 'this'
            Opcode::LoadLocal as u8,
            0,
            0,
            // Load y (param 2)
            Opcode::LoadLocal as u8,
            2,
            0,
            // Set this.y = y
            Opcode::StoreFieldExact as u8,
            1,
            0,
            // Return null (constructor doesn't return value)
            Opcode::ConstNull as u8,
            Opcode::Return as u8,
        ],
    };
    module.functions.push(constructor_fn);

    let result = vm.execute(&module).unwrap();
    assert_eq!(result, Value::i32(10));
}

#[test]
fn test_call_super() {
    // Test CALL_SUPER opcode (calling parent constructor)
    let mut vm = Vm::new();

    let mut module = Module::new("test".to_string());
    module.classes.push(class_def("Shape", 1, None));
    module.classes.push(class_def("Circle", 2, Some(0)));

    // Main function: creates Circle(5, "red")
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 1,
        code: vec![
            // Push constructor arguments (radius, color)
            Opcode::ConstI32 as u8,
            5,
            0,
            0,
            0, // radius
            Opcode::ConstI32 as u8,
            1, // Simplified: use 1 for "red"
            0,
            0,
            0,
            // CALL_CONSTRUCTOR class=1 (Circle), arg_count=2
            Opcode::CallConstructor as u8,
            1,
            0,
            0,
            0,
            2,
            0,
            // Store object
            Opcode::StoreLocal as u8,
            0,
            0,
            // Return field 1 (radius) to verify
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::LoadFieldExact as u8,
            1,
            0,
            Opcode::Return as u8,
        ],
    };
    module.functions.push(main_fn);

    // Shape constructor: sets color
    let shape_constructor = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "Shape::constructor".to_string(),
        param_count: 2, // this + color
        local_count: 2, // total locals = 2 (this + color)
        code: vec![
            // this.color = color
            Opcode::LoadLocal as u8,
            0,
            0, // this
            Opcode::LoadLocal as u8,
            1,
            0, // color
            Opcode::StoreFieldExact as u8,
            0,
            0, // field 0 (color)
            Opcode::ConstNull as u8,
            Opcode::Return as u8,
        ],
    };
    module.functions.push(shape_constructor);

    // Circle constructor: calls super, then sets radius
    let circle_constructor = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "Circle::constructor".to_string(),
        param_count: 3, // this + radius + color
        local_count: 3, // total locals = 3 (this + radius + color)
        code: vec![
            // Call super(color) - CALL_SUPER needs 'this' + args on stack
            Opcode::LoadLocal as u8,
            0,
            0, // this
            Opcode::LoadLocal as u8,
            2,
            0, // color
            // CALL_SUPER class=1 (Circle, which has Shape as parent), arg_count=1
            Opcode::CallSuper as u8,
            1,
            0,
            0,
            0, // current class 1 (Circle)
            1,
            0, // arg count 1 (just color)
            // Now set radius (field 1)
            Opcode::LoadLocal as u8,
            0,
            0, // this
            Opcode::LoadLocal as u8,
            1,
            0, // radius
            Opcode::StoreFieldExact as u8,
            1,
            0, // field 1 (radius)
            Opcode::ConstNull as u8,
            Opcode::Return as u8,
        ],
    };
    module.functions.push(circle_constructor);

    let result = vm.execute(&module).unwrap();
    assert_eq!(result, Value::i32(5)); // radius
}

#[test]
fn init_object_operand_is_a_field_offset_not_a_value_count() {
    // InitObject's u16 operand is the field offset to store into, not a count of
    // values to consume. Two ops with operands 0 and 1 must therefore land in
    // fields 0 and 1, and each op must consume exactly one value and leave the
    // object on the stack.
    //
    // This discriminates against the old "pop N values" contract: read as a
    // count, operand 0 would consume nothing and operand 1 would consume the
    // single value into a running cursor, so field 1 would not be 20.
    let mut vm = Vm::new();

    let mut module = Module::new("test".to_string());
    module.classes.push(class_def("Point", 2, None));
    let main_fn = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 1,
        code: vec![
            // new Point() -> local 0
            Opcode::NewType as u8,
            0,
            0, // class index 0
            Opcode::StoreLocal as u8,
            0,
            0,
            // obj.x = 10 via InitObject(field 0)
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ConstI32 as u8,
            10,
            0,
            0,
            0,
            Opcode::InitObject as u8,
            0,
            0, // field offset 0
            // obj.y = 20 via InitObject(field 1)
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::ConstI32 as u8,
            20,
            0,
            0,
            0,
            Opcode::InitObject as u8,
            1,
            0, // field offset 1
            // return obj.y; InitObject must have left the object on the stack
            // each time, so local 0 is still the receiver here.
            Opcode::LoadLocal as u8,
            0,
            0,
            Opcode::LoadFieldExact as u8,
            1,
            0, // field offset 1
            Opcode::Return as u8,
        ],
    };
    module.functions.push(main_fn);

    let result = vm.execute(&module).unwrap();
    assert_eq!(result, Value::i32(20));
}

// ---------------------------------------------------------------------------
// Descriptor accessor coverage (D4.3 / ALY-46)
//
// `Object.defineProperty` with a `get` descriptor stores `__node_compat_descriptor`
// metadata, and the interpreter's field handlers consult it and invoke the getter
// as a callable frame. The JIT field helpers never did, which is why five object
// opcodes were demoted to `Rejected`. These tests pin the interpreter behaviour
// the JIT has to match, and they are the corpus the plan requires before any of
// those opcodes is promoted again.
//
// No test in this suite previously exercised `defineProperty` at all.
// ---------------------------------------------------------------------------

/// Native id of `Object.defineProperty` (`compiler/native_id.rs`).
const OBJECT_DEFINE_PROPERTY: u16 = 0x0004;

/// Positional layout of a Node-compat property descriptor, matching
/// `legacy_field_index_for_layout` in `vm/interpreter/opcodes/objects.rs`:
/// index 4 is the getter and index 0 is the data value.
const DESCRIPTOR_FIELDS: [&str; 6] = [
    "value",
    "writable",
    "configurable",
    "enumerable",
    "get",
    "set",
];

/// Builds a module whose `main` (function 0) creates a target object, optionally
/// installs a `get` accessor on it via `Object.defineProperty`, then reads field
/// 0 and returns. Function 1 is the getter body, returning 42.
///
/// `with_descriptor` false leaves the descriptor uninstalled, which is the
/// control: field 0 is never assigned, so a raw read must yield null.
fn accessor_read_module(with_descriptor: bool) -> Module {
    use raya_engine::compiler::bytecode::{
        ClassReflectionData, FieldReflectionData, ReflectionData,
    };

    let mut module = Module::new("accessor".to_string());
    module.classes.push(class_def("Point", 1, None));
    // Descriptor layout is positional: `legacy_field_index_for_layout` maps
    // [value, writable, configurable, enumerable, get, set] to indices 0..5.
    // A short descriptor makes distinct probed names collide onto one slot, which
    // trips the "cannot mix accessors and value" guard.
    module.classes.push(class_def("Descriptor", 6, None));
    module.constants.strings.push("x".to_string());

    // Field names reach `class_metadata` only via reflection data, which is what
    // `field_name_for_offset` uses to map an offset back to a name.
    module.reflection = Some(ReflectionData {
        classes: vec![
            ClassReflectionData {
                fields: vec![FieldReflectionData {
                    name: "x".to_string(),
                    type_name: "i32".to_string(),
                    is_readonly: false,
                    is_static: false,
                }],
                method_names: vec![],
                static_field_names: vec![],
            },
            ClassReflectionData {
                fields: DESCRIPTOR_FIELDS
                    .iter()
                    .map(|name| FieldReflectionData {
                        name: (*name).to_string(),
                        type_name: "func".to_string(),
                        is_readonly: false,
                        is_static: false,
                    })
                    .collect(),
                method_names: vec![],
                static_field_names: vec![],
            },
        ],
    });

    // The getter, referenced by `MakeClosure`.
    module.functions.push(Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "getter".to_string(),
        param_count: 0,
        local_count: 0,
        code: vec![Opcode::ConstI32 as u8, 42, 0, 0, 0, Opcode::Return as u8],
    });

    fn push_local(code: &mut Vec<u8>, slot: u16) {
        code.push(Opcode::LoadLocal as u8);
        code.extend_from_slice(&slot.to_le_bytes());
    }
    fn store_local(code: &mut Vec<u8>, slot: u16) {
        code.push(Opcode::StoreLocal as u8);
        code.extend_from_slice(&slot.to_le_bytes());
    }

    let mut code: Vec<u8> = Vec::new();

    // local 0 = target
    code.push(Opcode::NewType as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    store_local(&mut code, 0);

    if with_descriptor {
        // local 1 = descriptor object
        code.push(Opcode::NewType as u8);
        code.extend_from_slice(&1u16.to_le_bytes());
        store_local(&mut code, 1);
        // local 2 = getter closure over function 1, no captures
        code.push(Opcode::MakeClosure as u8);
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&0u16.to_le_bytes());
        store_local(&mut code, 2);
        // descriptor.get = closure  (InitObject peeks the object, pops the value).
        // Index 4 is "get" in the descriptor layout.
        push_local(&mut code, 1);
        push_local(&mut code, 2);
        code.push(Opcode::InitObject as u8);
        code.extend_from_slice(&4u16.to_le_bytes());
        // InitObject leaves the descriptor on the stack; drop it so the final
        // Return sees only the field value.
        code.push(Opcode::Pop as u8);
        // Object.defineProperty(target, "x", descriptor). NativeCall's operand is
        // (u16 nativeId, u8 argCount) -- omitting the count makes the following
        // opcode byte read as the count and underflow the stack.
        push_local(&mut code, 0);
        code.push(Opcode::ConstStr as u8);
        code.extend_from_slice(&0u16.to_le_bytes());
        push_local(&mut code, 1);
        code.push(Opcode::NativeCall as u8);
        code.extend_from_slice(&OBJECT_DEFINE_PROPERTY.to_le_bytes());
        code.push(3u8);
    }

    // return target.x
    push_local(&mut code, 0);
    code.push(Opcode::LoadFieldExact as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    code.push(Opcode::Return as u8);

    module.functions.insert(
        0,
        Function {
            signature_id: 0,
            local_types: Vec::new(),
            abi_version: 1,
            name: "main".to_string(),
            param_count: 0,
            local_count: 3,
            code,
        },
    );
    module
}

#[test]
fn load_field_exact_without_descriptor_reads_the_raw_field() {
    // Control for the test below. Field 0 is never assigned, so a plain read is
    // null. This is what the JIT helper does, unconditionally.
    let mut vm = Vm::new();
    let result = vm.execute(&accessor_read_module(false)).unwrap();
    assert!(result.is_null(), "expected null raw field, got {result:?}");
}

#[test]
fn load_field_exact_invokes_a_descriptor_getter() {
    // With a `get` accessor installed, `LoadFieldExact` must invoke the getter and
    // return 42 -- not the raw slot, which is still null.
    let mut vm = Vm::new();
    let result = vm.execute(&accessor_read_module(true)).unwrap();
    assert_eq!(
        result,
        Value::i32(42),
        "descriptor getter was not invoked; got {result:?}"
    );
}

/// Builds a module where `main` (function 0) optionally installs a `set`
/// accessor on field 0 ("x") of a two-field target, stores 99 into it, then reads
/// field 1 ("sink"). Function 1 is the setter, which captures the target and writes
/// the constant 7 into the sink field.
///
/// Field 1 exists so the setter has somewhere to write that is *not* itself
/// accessor-backed: a setter writing field 0 would re-enter the setter.
///
/// Returns `sink`. With the accessor, the setter runs and the sink is 7. Without
/// it, the store lands in field 0 and the sink is never touched, so it is null.
/// The stored value (99) therefore never appears, which is what makes the pair
/// discriminating rather than merely asserting two different constants.
fn accessor_setter_module(with_descriptor: bool) -> Module {
    use raya_engine::compiler::bytecode::{
        ClassReflectionData, FieldReflectionData, ReflectionData,
    };

    fn push_local(code: &mut Vec<u8>, slot: u16) {
        code.push(Opcode::LoadLocal as u8);
        code.extend_from_slice(&slot.to_le_bytes());
    }
    fn store_local(code: &mut Vec<u8>, slot: u16) {
        code.push(Opcode::StoreLocal as u8);
        code.extend_from_slice(&slot.to_le_bytes());
    }

    let mut module = Module::new("accessor_setter".to_string());
    module.classes.push(class_def("Holder", 2, None));
    module.classes.push(class_def("Descriptor", 6, None));
    module.constants.strings.push("x".to_string());

    module.reflection = Some(ReflectionData {
        classes: vec![
            ClassReflectionData {
                fields: vec![
                    FieldReflectionData {
                        name: "x".to_string(),
                        type_name: "i32".to_string(),
                        is_readonly: false,
                        is_static: false,
                    },
                    FieldReflectionData {
                        name: "sink".to_string(),
                        type_name: "i32".to_string(),
                        is_readonly: false,
                        is_static: false,
                    },
                ],
                method_names: vec![],
                static_field_names: vec![],
            },
            ClassReflectionData {
                fields: DESCRIPTOR_FIELDS
                    .iter()
                    .map(|name| FieldReflectionData {
                        name: (*name).to_string(),
                        type_name: "func".to_string(),
                        is_readonly: false,
                        is_static: false,
                    })
                    .collect(),
                method_names: vec![],
                static_field_names: vec![],
            },
        ],
    });

    // The setter: capture 0 is the target; write the constant 7 into the sink
    // field. It deliberately ignores its argument, so a value that comes back as
    // 7 rather than 99 can only have come through the setter.
    module.functions.push(Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "setter".to_string(),
        param_count: 1,
        local_count: 0,
        code: vec![
            Opcode::LoadCaptured as u8,
            0,
            0, // capture 0: the target
            Opcode::ConstI32 as u8,
            7,
            0,
            0,
            0,
            Opcode::StoreFieldExact as u8,
            1,
            0, // sink field
            Opcode::Return as u8,
        ],
    });

    let mut code: Vec<u8> = Vec::new();
    // local 0 = target
    code.push(Opcode::NewType as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    store_local(&mut code, 0);

    if with_descriptor {
        // local 1 = descriptor object
        code.push(Opcode::NewType as u8);
        code.extend_from_slice(&1u16.to_le_bytes());
        store_local(&mut code, 1);
        // local 2 = setter closure capturing the target
        push_local(&mut code, 0);
        code.push(Opcode::MakeClosure as u8);
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&1u16.to_le_bytes()); // captureCount = 1
        store_local(&mut code, 2);
        // descriptor.set = closure; index 5 is "set"
        push_local(&mut code, 1);
        push_local(&mut code, 2);
        code.push(Opcode::InitObject as u8);
        code.extend_from_slice(&5u16.to_le_bytes());
        code.push(Opcode::Pop as u8);
        // Object.defineProperty(target, "x", descriptor)
        push_local(&mut code, 0);
        code.push(Opcode::ConstStr as u8);
        code.extend_from_slice(&0u16.to_le_bytes());
        push_local(&mut code, 1);
        code.push(Opcode::NativeCall as u8);
        code.extend_from_slice(&OBJECT_DEFINE_PROPERTY.to_le_bytes());
        code.push(3u8);
    }

    // target.x = 99. With the accessor this invokes the setter, which writes 7
    // into the sink; without it the 99 lands in field 0.
    push_local(&mut code, 0);
    code.push(Opcode::ConstI32 as u8);
    code.extend_from_slice(&99i32.to_le_bytes());
    code.push(Opcode::StoreFieldExact as u8);
    code.extend_from_slice(&0u16.to_le_bytes());

    // return the sink
    push_local(&mut code, 0);
    code.push(Opcode::LoadFieldExact as u8);
    code.extend_from_slice(&1u16.to_le_bytes());
    code.push(Opcode::Return as u8);

    module.functions.insert(
        0,
        Function {
            signature_id: 0,
            local_types: Vec::new(),
            abi_version: 1,
            name: "main".to_string(),
            param_count: 0,
            local_count: 3,
            code,
        },
    );
    module
}

#[test]
fn store_field_without_descriptor_writes_the_field_directly() {
    // Control for the test below. No descriptor, so the 99 lands in field 0 and
    // the sink is never written.
    let mut vm = Vm::new();
    let result = vm.execute(&accessor_setter_module(false)).unwrap();
    assert!(
        result.is_null(),
        "sink should be untouched without a setter, got {result:?}"
    );
}

#[test]
fn store_field_invokes_a_descriptor_setter() {
    // With a `set` accessor, the store must invoke the setter. The setter ignores
    // its argument and writes 7, so seeing 7 proves the accessor ran and seeing 99
    // would mean the raw field was written instead.
    let mut vm = Vm::new();
    let result = vm.execute(&accessor_setter_module(true)).unwrap();
    assert_eq!(
        result,
        Value::i32(7),
        "descriptor setter was not invoked; got {result:?}"
    );
}

/// `builtin::reflect::CREATE_PROXY` (`vm/builtin.rs:544`). `createProxy` is
/// dispatched from `CallMethodExact`, so a proxy can be created from bytecode --
/// no local injection needed.
const CREATE_PROXY: u32 = 0x0DB0;

/// Builds a module whose `main` creates a target with `x = 42`, wraps it in a
/// proxy via `createProxy(target, handler)`, then reads field 0 **through the
/// proxy** and returns it.
///
/// `CallMethodExact` pops exactly `arg_count` arguments and leaves the receiver on
/// the stack, so the bytecode pops the receiver before storing the proxy.
///
/// The expected answer is 42, and it can only come through the target: the proxy
/// object itself has no fields, and `ensure_object_receiver` would reject a proxy
/// outright if the handler did not unwrap it first. This pins the interpreter
/// contract the JIT helpers break -- `jit_object_ptr_checked` returns `None` for a
/// proxy where the interpreter unwraps to the target.
fn proxy_field_read_module() -> Module {
    use raya_engine::compiler::bytecode::{
        ClassReflectionData, FieldReflectionData, ReflectionData,
    };

    fn push_local(code: &mut Vec<u8>, slot: u16) {
        code.push(Opcode::LoadLocal as u8);
        code.extend_from_slice(&slot.to_le_bytes());
    }
    fn store_local(code: &mut Vec<u8>, slot: u16) {
        code.push(Opcode::StoreLocal as u8);
        code.extend_from_slice(&slot.to_le_bytes());
    }

    let mut module = Module::new("proxy_field_read".to_string());
    module.classes.push(class_def("Target", 1, None));
    module.classes.push(class_def("Handler", 0, None));

    module.reflection = Some(ReflectionData {
        classes: vec![
            ClassReflectionData {
                fields: vec![FieldReflectionData {
                    name: "x".to_string(),
                    type_name: "i32".to_string(),
                    is_readonly: false,
                    is_static: false,
                }],
                method_names: vec![],
                static_field_names: vec![],
            },
            ClassReflectionData {
                fields: vec![],
                method_names: vec![],
                static_field_names: vec![],
            },
        ],
    });

    let mut code: Vec<u8> = Vec::new();
    // local 0 = target with x = 42
    code.push(Opcode::NewType as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    store_local(&mut code, 0);
    push_local(&mut code, 0);
    code.push(Opcode::ConstI32 as u8);
    code.extend_from_slice(&42i32.to_le_bytes());
    code.push(Opcode::InitObject as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    code.push(Opcode::Pop as u8);

    // local 1 = handler, also used as the ignored CallMethodExact receiver
    code.push(Opcode::NewType as u8);
    code.extend_from_slice(&1u16.to_le_bytes());
    store_local(&mut code, 1);

    // createProxy(target, handler)
    push_local(&mut code, 1);
    push_local(&mut code, 0);
    push_local(&mut code, 1);
    code.push(Opcode::CallMethodExact as u8);
    code.extend_from_slice(&CREATE_PROXY.to_le_bytes());
    code.extend_from_slice(&2u16.to_le_bytes());
    // CallMethodExact pops 2 args and pushes the proxy, so the stack is now
    // [receiver, proxy]. Store the proxy (top) first, then drop the receiver --
    // popping first would discard the proxy and leave the 0-field handler in
    // local 2, which reads back as null.
    store_local(&mut code, 2);
    code.push(Opcode::Pop as u8);

    // return proxy.x
    push_local(&mut code, 2);
    code.push(Opcode::LoadFieldExact as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    code.push(Opcode::Return as u8);

    module.functions.push(Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 3,
        code,
    });
    module
}

#[test]
fn field_access_through_a_proxy_currently_raises_and_that_is_a_defect() {
    // Characterization test, not an endorsement.
    //
    // Field access on a proxy does NOT work today, in either engine:
    //
    //   * the interpreter raises `TypeError: Expected Object receiver for field
    //     access, got UnknownGcType`, because `ensure_object_receiver`
    //     (objects.rs:459) has no Proxy case and runs BEFORE the handler's
    //     `unwrap_proxy_target` call. That makes the unwrap unreachable -- the
    //     eight `unwrap_proxy_target` sites in the field handlers are dead code
    //     for proxies.
    //   * the JIT helpers never unwrap at all; `jit_object_ptr_checked` returns
    //     `None` for a proxy and the load yields null.
    //
    // So this is a uniformly unsupported feature rather than an interpreter/JIT
    // divergence, and it needs a design decision before it can be "fixed":
    // silently unwrapping bypasses the proxy handler, and `objects.rs:553` carries
    // a TODO saying full trap support would call `handler.get(target, fieldName)`.
    // Whether an unwrapped proxy should bypass traps or raise is not this
    // milestone's call.
    //
    // When that decision lands, this test flips: either proxy field access starts
    // working (expect 42) or the TypeError becomes deliberate and this becomes a
    // documented error contract.
    let mut vm = Vm::new();
    let outcome = vm.execute(&proxy_field_read_module());
    match outcome {
        Ok(value) => panic!("expected the current TypeError, got Ok({value:?})"),
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("Expected Object receiver for field access"),
                "expected the proxy receiver TypeError, got: {message}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Centralized checked object field mutation (D4.3 scope item 4)
//
// `Object::set_field` returned `Result<(), String>`, so every caller had to
// format or match on a message to tell a bounds failure from a binding failure.
// `checked_set_field` is the single checked path, mirroring the `checked_set` /
// `checked_push` family D4.2 introduced for arrays, and `set_field` now delegates
// to it so the ~48 existing call sites keep byte-identical behaviour.
// ---------------------------------------------------------------------------

#[test]
fn checked_set_field_reports_a_typed_bounds_error() {
    use raya_engine::vm::object::ObjectFieldStoreError;

    let mut object = raya_engine::vm::object::Object::new_nominal(1, 0, 2);
    assert_eq!(object.field_count(), 2);

    // In bounds: stores, and reports no error.
    assert!(object.checked_set_field(1, Value::i32(5)).is_ok());
    assert_eq!(object.get_field(1), Some(Value::i32(5)));

    // Out of bounds: a typed, inspectable variant carrying the numbers, not a
    // string that has to be matched on.
    let error = object
        .checked_set_field(7, Value::i32(9))
        .expect_err("offset 7 is outside a 2-field object");
    assert_eq!(
        error,
        ObjectFieldStoreError::OutOfBounds {
            index: 7,
            field_count: 2
        }
    );

    // The failed store must not have mutated anything. Object fields are
    // null-initialised, so the untouched field reads as null rather than absent.
    assert_eq!(object.field_count(), 2);
    assert_eq!(object.get_field(0), Some(Value::null()));
    assert_eq!(object.get_field(1), Some(Value::i32(5)));
}

#[test]
fn set_field_message_is_unchanged_by_the_delegation() {
    // Regression guard for the ~48 call sites still using `set_field`: the wrapper
    // must render exactly the message the original implementation produced,
    // byte for byte. If this drifts, error output changes across the whole VM.
    let mut object = raya_engine::vm::object::Object::new_nominal(1, 0, 1);
    assert_eq!(
        object.set_field(4, Value::i32(1)).unwrap_err(),
        "Field index 4 out of bounds (object has 1 fields)"
    );
    // And the success path is unchanged too.
    assert!(object.set_field(0, Value::i32(3)).is_ok());
    assert_eq!(object.get_field(0), Some(Value::i32(3)));
}

// ---------------------------------------------------------------------------
// Interpreter RefCell baseline (D4.4)
//
// This is the *interpreter* half of the RefCell story. The three JIT lowering arms
// have direct-lift tests, but until this existed there was nothing to compare them
// against: no test exercised the interpreter's `NewRefCell`/`LoadRefCell`/
// `StoreRefCell` handlers at all. A promotion justified by comparing the JIT only
// against itself is not evidence, so this has to come first.
//
// It is also the same gap that let D4.3 ship: the accessor divergence survived
// because no test in the suite constructed a `defineProperty` case.
//
// Note the weak type check these handlers use -- `is_ptr()` only, never a
// GC-header TypeId (ALY-54). A non-RefCell heap value is therefore accepted and
// reinterpreted, here as in the JIT helpers. These tests pin the *actual*
// behaviour, not the behaviour anyone would want.

/// `NewRefCell` pops an initial value and pushes a new cell; `LoadRefCell` pops a
/// cell and pushes its contents; `StoreRefCell` pops a value then a cell and pushes
/// nothing (net -2).
#[test]
fn refcell_opcodes_roundtrip_through_the_interpreter() {
    let mut vm = Vm::new();

    let mut code: Vec<u8> = Vec::new();
    // cell = new RefCell(11); push it back so we can store through it.
    code.push(Opcode::ConstI32 as u8);
    code.extend_from_slice(&11i32.to_le_bytes());
    code.push(Opcode::NewRefCell as u8);
    // cell.x = 99  (StoreRefCell pops value then cell, pushes nothing)
    //
    // The Dup must come BEFORE the value: it duplicates the top of stack, which is
    // the cell at that point. Duplicating after pushing 99 would copy the value,
    // and the store would then target that immediate -- which the interpreter's
    // is_ptr() check rejects.
    code.push(Opcode::Dup as u8); // [cell, cell]
    code.push(Opcode::ConstI32 as u8);
    code.extend_from_slice(&99i32.to_le_bytes()); // [cell, cell, 99]
    code.push(Opcode::StoreRefCell as u8); // -> [cell], cell now holds 99
    // return cell's contents
    code.push(Opcode::LoadRefCell as u8);
    code.push(Opcode::Return as u8);

    let mut module = Module::new("refcell".to_string());
    module.functions.push(Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 0,
        code,
    });

    let result = vm.execute(&module).unwrap();
    assert_eq!(
        result,
        Value::i32(99),
        "StoreRefCell must write through to the cell that LoadRefCell reads"
    );
}

#[test]
fn refcell_opcodes_reject_a_non_pointer_receiver() {
    let mut vm = Vm::new();

    let mut code: Vec<u8> = Vec::new();
    code.push(Opcode::ConstI32 as u8);
    code.extend_from_slice(&7i32.to_le_bytes());
    // LoadRefCell on an immediate: the is_ptr() check must reject it.
    code.push(Opcode::LoadRefCell as u8);
    code.push(Opcode::Return as u8);

    let mut module = Module::new("refcell_bad_receiver".to_string());
    module.functions.push(Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 0,
        code,
    });

    let error = vm
        .execute(&module)
        .expect_err("a non-pointer RefCell receiver must be a TypeError");
    let message = error.to_string();
    assert!(
        message.contains("Expected RefCell"),
        "expected the interpreter's RefCell TypeError, got: {message}"
    );
}


// ---------------------------------------------------------------------------
// Interpreter closure baseline (D4.4)
//
// Closures are the first family on this milestone whose baseline could not be
// written from bytecode alone. `Closure.captures` is read only by the GC tracer,
// the snapshot codec, `get_captured` and `capture_count` — the sole bytecode-level
// reader is `LoadCaptured`, which needs an active closure. So `MakeClosure` pushes
// a closure nobody can inspect, and `SetClosureCapture` patches a slot inside a
// closure nobody can read. Their only observable effect is through code running
// *inside* that closure.
//
// The missing piece of that harness is `Call` with the sentinel function index:
//
//     if func_index == 0xFFFFFFFF { /* closure call */ }
//
// which lifts the closure value from beneath the arguments and invokes it. That is
// what these two tests use. `tests/closure_tests/mod.rs` holds 26 tests of which 12
// are `assert!(true)` placeholders and 14 exercise *Rust* closures, so it is not
// coverage; these live here instead, next to the object-model baselines.

/// `Opcode::Call` selects a closure call with this function index, lifting the
/// closure value from beneath the arguments (`vm/interpreter/opcodes/calls.rs`).
const CLOSURE_CALL: u32 = 0xFFFF_FFFF;

/// Builds a two-function module: `main` at index 0 and a closure body at index 1
/// made of `body`. `main_body` is appended after the `MakeClosure` and is
/// responsible for the `Call` and the `Return`.
fn closure_call_module(initial_capture: i32, body: &[u8], main_body: &[u8]) -> Module {
    let mut module = Module::new("closure_call".to_string());

    let mut closure_code: Vec<u8> = body.to_vec();
    closure_code.push(Opcode::Return as u8);
    module.functions.push(Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "closure_body".to_string(),
        param_count: 0,
        local_count: 0,
        code: closure_code,
    });

    let mut code: Vec<u8> = Vec::new();
    code.push(Opcode::ConstI32 as u8);
    code.extend_from_slice(&initial_capture.to_le_bytes());
    code.push(Opcode::MakeClosure as u8);
    code.extend_from_slice(&1u32.to_le_bytes()); // func_index = closure_body
    code.extend_from_slice(&1u16.to_le_bytes()); // capture_count = 1
    code.extend_from_slice(main_body);
    code.push(Opcode::Return as u8);
    module.functions.insert(
        0,
        Function {
            signature_id: 0,
            local_types: Vec::new(),
            abi_version: 1,
            name: "main".to_string(),
            param_count: 0,
            local_count: 0,
            code,
        },
    );
    module
}

#[test]
fn make_closure_then_call_invokes_the_closure_body() {
    // `LoadCaptured 0; Return` — the body just hands the capture back.
    let mut vm = Vm::new();
    let module = closure_call_module(
        42,
        &[Opcode::LoadCaptured as u8, 0, 0],
        &[
            Opcode::Call as u8,
            CLOSURE_CALL.to_le_bytes()[0],
            CLOSURE_CALL.to_le_bytes()[1],
            CLOSURE_CALL.to_le_bytes()[2],
            CLOSURE_CALL.to_le_bytes()[3],
            0,
            0, // arg_count = 0
        ],
    );
    let result = vm.execute(&module).expect("closure call must run");
    assert_eq!(
        result,
        Value::i32(42),
        "the closure body must observe the captured value"
    );
}

#[test]
fn set_closure_capture_patches_what_the_body_sees() {
    // Create a closure capturing 0, patch slot 0 to 42 through SetClosureCapture,
    // then call it. If the patch did not happen the body would see 0, so this is
    // discriminating rather than a test that merely constructs a closure.
    let mut vm = Vm::new();

    let mut main_tail: Vec<u8> = Vec::new();
    // SetClosureCapture pops the value then the closure, and pushes the closure back.
    // Dup BEFORE the value: it duplicates the top of stack, which is the closure at
    // that point. Duplicating after would copy the 42, and the opcode would then
    // take an immediate as its closure operand -- which the interpreter's is_ptr()
    // check rejects with "Expected closure", which is exactly what it did.
    main_tail.push(Opcode::Dup as u8); // [closure, closure]
    main_tail.push(Opcode::ConstI32 as u8);
    main_tail.extend_from_slice(&42i32.to_le_bytes()); // [closure, closure, 42]
    main_tail.push(Opcode::SetClosureCapture as u8);
    main_tail.extend_from_slice(&0u16.to_le_bytes()); // capture index 0
    main_tail.push(Opcode::Call as u8);
    main_tail.extend_from_slice(&CLOSURE_CALL.to_le_bytes());
    main_tail.extend_from_slice(&0u16.to_le_bytes()); // arg_count = 0

    let module = closure_call_module(0, &[Opcode::LoadCaptured as u8, 0, 0], &main_tail);
    let result = vm.execute(&module).expect("closure call must run");
    assert_eq!(
        result,
        Value::i32(42),
        "SetClosureCapture must be visible to the closure body; 0 means it never landed"
    );
}


/// `LoadCaptured` and `StoreCaptured` need a baseline shape nothing else on this
/// milestone did: they are only meaningful **inside** a closure body. A flat
/// bytecode program cannot observe them, because a closure's captures are read
/// solely through `LoadCaptured`, which needs an active closure, and the only way
/// to have one is to call into the closure.
///
/// So the module is `main` plus a body that stores its argument into its own
/// capture and reads it back:
///
/// ```text
/// main:  ConstI32 7; MakeClosure func=1 captures=[7]; ConstI32 42;
///        Call 0xFFFFFFFF arg_count=1; Return
/// body:  StoreCaptured 0; LoadCaptured 0; Return
/// ```
///
/// Expecting 42 rather than the captured 7 is what makes it discriminating: if
/// `StoreCaptured` did not land, the body would read back 7, and if
/// `LoadCaptured` were broken it would not observe the store at all.
#[test]
fn load_and_store_captured_are_observable_only_inside_a_closure() {
    use raya_engine::vm::interpreter::Vm;

    let mut module = Module::new("captured_baseline".to_string());

    // functions[0] is main; the VM's entry point must be named "main".
    module.functions.push(Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "closure_body".to_string(),
        param_count: 1,
        local_count: 1,
        code: vec![
            Opcode::StoreCaptured as u8, 0, 0, // capture 0 <- local 0 (the argument)
            Opcode::LoadCaptured as u8, 0, 0, // push capture 0 back
            Opcode::Return as u8,
        ],
    });

    let mut main_code: Vec<u8> = Vec::new();
    main_code.push(Opcode::ConstI32 as u8);
    main_code.extend_from_slice(&7i32.to_le_bytes()); // initial capture
    main_code.push(Opcode::MakeClosure as u8);
    main_code.extend_from_slice(&1u32.to_le_bytes()); // func_index = closure_body
    main_code.extend_from_slice(&1u16.to_le_bytes()); // capture_count = 1
    main_code.push(Opcode::ConstI32 as u8);
    main_code.extend_from_slice(&42i32.to_le_bytes()); // the argument
    main_code.push(Opcode::Call as u8);
    main_code.extend_from_slice(&CLOSURE_CALL.to_le_bytes());
    main_code.extend_from_slice(&1u16.to_le_bytes()); // arg_count = 1
    main_code.push(Opcode::Return as u8);

    module.functions.insert(
        0,
        Function {
            signature_id: 0,
            local_types: Vec::new(),
            abi_version: 1,
            name: "main".to_string(),
            param_count: 0,
            local_count: 0,
            code: main_code,
        },
    );

    let mut vm = Vm::new();
    let result = vm.execute(&module).expect("captured baseline must run");
    assert_eq!(
        result,
        Value::i32(42),
        "StoreCaptured must write the argument into the capture and LoadCaptured \
         must read it back; 7 means the store never landed, null means neither \
         opcode reached the closure's own captures"
    );
}


// ---------------------------------------------------------------------------
// BindMethod interpreter baseline (D4.4)
//
// The last piece of evidence BindMethod needs. Its handler has five ordered checks
// and three error classes, so the baseline has to pin both the success and the
// most specific failure — the structural-object case, which is the one a nominal
// test would never reach.
// ---------------------------------------------------------------------------

#[test]
fn bind_method_on_a_nominal_object_succeeds() {
    let mut vm = Vm::new();

    let mut module = Module::new("bind_method".to_string());
    // `register_classes` populates the vtable from `ClassDef::methods`, so the
    // class must DECLARE the method; a bare function pushed onto the module is not
    // enough, and slot 0 comes back as "Invalid method slot".
    let mut point = class_def("Point", 1, None);
    point.methods.push(raya_engine::compiler::bytecode::module::Method {
        name: "get".to_string(),
        function_id: 0,
        slot: 0,
    });
    module.classes.push(point);
    // The method body, at the slot the class declares.
    module.functions.push(Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "get".to_string(),
        param_count: 1,
        local_count: 1,
        code: vec![Opcode::ConstI32 as u8, 7, 0, 0, 0, Opcode::Return as u8],
    });

    let mut main_code: Vec<u8> = Vec::new();
    main_code.push(Opcode::NewType as u8);
    main_code.extend_from_slice(&0u16.to_le_bytes()); // class 0 = Point
    main_code.push(Opcode::BindMethod as u8);
    main_code.extend_from_slice(&0u16.to_le_bytes()); // method slot 0
    main_code.push(Opcode::Return as u8);
    module.functions.insert(
        0,
        Function {
            signature_id: 0,
            local_types: Vec::new(),
            abi_version: 1,
            name: "main".to_string(),
            param_count: 0,
            local_count: 0,
            code: main_code,
        },
    );

    let result = vm
        .execute(&module)
        .expect("binding a method on a nominal object must succeed");
    assert!(
        result.is_ptr(),
        "BindMethod must produce a heap BoundMethod, got {result:?}"
    );
}

#[test]
fn bind_method_on_a_structural_object_is_a_type_error() {
    // The handler checks `nominal_type_id_usize()` and refuses a structural
    // object with "Cannot bind method on structural object value". A nominal-only
    // baseline would never reach this branch, so it is pinned separately.
    let mut vm = Vm::new();

    let mut code: Vec<u8> = Vec::new();
    // ObjectLiteral with no fields makes a structural object.
    code.push(Opcode::ObjectLiteral as u8);
    code.extend_from_slice(&0u32.to_le_bytes()); // layout id
    code.extend_from_slice(&0u16.to_le_bytes()); // field count
    code.push(Opcode::BindMethod as u8);
    code.extend_from_slice(&0u16.to_le_bytes()); // method slot 0
    code.push(Opcode::Return as u8);

    let mut module = Module::new("bind_structural".to_string());
    module.functions.push(Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "main".to_string(),
        param_count: 0,
        local_count: 0,
        code,
    });

    let error = vm
        .execute(&module)
        .expect_err("a structural object must not accept BindMethod");
    let message = error.to_string();
    assert!(
        message.contains("structural") || message.contains("method binding"),
        "expected the handler's structural-object or receiver diagnostic, got: {message}"
    );
}
