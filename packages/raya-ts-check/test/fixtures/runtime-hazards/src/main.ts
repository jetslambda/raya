export class Point {
  x: number = 0;
  y: number = 0;
}

// RT2003 line 8
(Point.prototype as any).origin = () => new Point();

// RT2004 line 11
export function runDynamic(src: string): void {
  eval(src);
}

// RT2004 line 15
export function makeFn(body: string): Function {
  return new Function("x", body);
}

// RT2002 line 19
export function lookup(obj: Record<string, number>, key: string): number {
  return obj[key];
}

// RT2001 line 23
export function scale(v: number, n: number): number {
  let total = v;
  for (let i = 0; i < n; i++) {
    total = total * v + i - 1;
  }
  return total;
}
