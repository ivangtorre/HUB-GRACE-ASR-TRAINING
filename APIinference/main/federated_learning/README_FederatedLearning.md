## Federated learning

This README integrates and explain NVIDIA Flare integration for the Federated Learning. 

This folder may have some duplicated code. However, the training scripts for the not Federated Learning are 
still not provided in this repository.

FL application needs a certain structure:
  - the "custom" folder with all the code
  - the "config" folder with json configuration for nvidia-flare

In order to achieve it in a clean a repeatable way, I think that the best approach is to create a
small automation script to take care of copying all that is needed and create the required stucture.

That implies:
  - create the app_folder (for the FL), with a configurable name (e.g. nerc_tool, or whatever)
  - create the "custom" folder (with that very name, as indicated in the instructions)
  - copy the whole top-level package "grace_sep_labelling" to "custom"
  - create a "config" folder next to "custom"
  - copy the predefined config files to the "config" folder

#And the execution of that script would be something like:

```bash
prepare_fl_app output_path [other_opts]
```