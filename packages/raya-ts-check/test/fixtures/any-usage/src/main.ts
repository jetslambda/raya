export interface User {
  name: string;
  age: number;
}

export function greet(name: any): string {
  return "hi " + name;
}

export function load(body: string): User {
  return JSON.parse(body) as User;
}

export function first(xs: string[]): string {
  return xs[0]!;
}

export function sum(a, b) {
  return a + b;
}

export const loose = JSON.parse("{}");
