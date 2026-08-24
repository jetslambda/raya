// Syntax the Raya parser rejects today: decorators, constructor
// parameter properties, and ambient declarations.

function frozen(target: Function): Function {
  return target;
}

@frozen
class Gatekeeper {
  private attempts: number;

  constructor(private readonly label: string) {
    this.attempts = 0;
  }
}

declare global {
  interface Window {
    rayaReady?: boolean;
  }
}

export { Gatekeeper };
