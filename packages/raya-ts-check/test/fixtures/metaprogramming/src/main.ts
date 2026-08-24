const settings = { theme: "dark", timeout: 30 };

const tracked = new Proxy(settings, {});

const theme = Reflect.get(settings, "theme");
Reflect.set(settings, "timeout", 60);
Reflect.has(settings, "theme");

export { tracked, theme };
