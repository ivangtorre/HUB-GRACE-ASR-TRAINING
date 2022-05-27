# ASR and KWS Federated Learning instructions

## Initial setup
```
sudo apt-get install python3-venv
```

1. Create a virtual environment with:

`python3 -m venv nvflare-env
`

python3.8 -m install requirements.txt


2. Activate the virtualenv by running the following command:

`source nvflare-env/bin/activate`

## Prepare the models to be deployed
To get started with a proof of concept (POC) setup after Installation, run this command to generates a poc folder with a server, two clients, and one admin:
```
poc -n 2
```
Copy the files to the working folder (upload folder for the admin):
```
mkdir -p poc/admin/transfer
cp -rf HUB-GRACE-ASR-TRAINING/APIinference/federatedlearning/* poc/admin/transfer
```

## Setup the environment
Open four different consoles and run each command on the respective console. We therefore have to start the server first:
```
./poc/server/startup/start.sh
```
Open a new terminal and start the first client:
```
./poc/site-1/startup/start.sh
```
Open another terminal and start the second client:
```
./poc/site-2/startup/start.sh
```
In one last terminal, start the admin:
```
./poc/admin/startup/fl_admin.sh localhost    
```

## Launch the federated learning
With the admin client command prompt successfully connected and logged in, enter the commands below in order. 
Pay close attention to what happens in each of four terminals. 
You can see how the admin controls the server and clients with each command.

```
upload_app hello-pt
```

Uploads the application from the admin client to the server’s staging area.
```
set_run_number 1
```

Creates a run directory in the workspace for the run_number on the server and all clients. The run directory allows for the isolation of different runs so the information in one particular run does not interfere with other runs.
```
deploy_app hello-pt all
```

This will make the hello-pt application the active one in the run_number workspace. After the above two commands, the server and all the clients know the hello-pt application will reside in the run_1 workspace.
```
start_app all
```

This start_app command instructs the NVIDIA FLARE server and clients to start training with the hello-pt application in that run_1 workspace.
From time to time, you can issue check_status server in the admin client to check the entire training progress.
You should now see how the training does in the very first terminal (the one that started the server).
Once the fl run is complete and the server has successfully aggregated the clients’ results after all the rounds, run the following commands in the fl_admin to shutdown the system (while inputting admin when prompted with user name):

```
shutdown client
shutdown server
```








