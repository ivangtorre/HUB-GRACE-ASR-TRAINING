docker-compose2 up --detach
echo "Please wait..."
sleep 8
STR=`ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | awk '{print $1; exit}'`
echo "ASR and KWS is now running in:" $STR":5050"

