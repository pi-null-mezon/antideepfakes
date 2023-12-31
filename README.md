Deepfake detection tool 
===

SystemFailure© solution for challenge'2023

![](./figures/logo.png)

## Architecture

![](./figures/antideepfake.png)

---

- [singleshot](./singleshot) - tools to learn single shot deepfake detector that works on single photo

- [sequence](./sequence) - tools to learn deepfake detector that analyze sequences of frames (with frozen single shot backbone)

- [sequence++](./sequence++) - tools to learn advanced deepfake detector that analyze sequences of frames

- [test](./test) - final test of trained models before and after tracing

- [API](./api) - API and docker files to build stand-alone service

- [scripts](./scripts) - scripts for the supervisors (to process local folder via API)

---

*designed by SystemFailure©, november 2023*