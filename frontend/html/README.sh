#install apache and PHP
sudo apt update
sudo apt install apache2
sudo apt install php libapache2-mod-php

#create json
base64 TESTING/spanish_audio2.wav -w 0 | sed 's/\//%2F/g' | sed 's/\+/%2B/g' | sed 's/=/%3D/g' > TESTING/spanish_audio2.txt
cat TESTING/spanish_audio2.txt | sed 's/^\(.*\)$/language=es\&keywords=\&audio=\1/' > TESTING/spanish_audio2.request

#ASR (skynet)
curl \
-X POST \
--data "@TESTING/spanish_audio2.request" \
http://192.168.25.16:8421/json/

#ASR (speech)
curl \
-X POST \
--data "@TESTING/spanish_audio2.request" \
http://192.168.25.7:8421/json/

#http://speech/GRACE/index.html <=> https://grace.speech.vicomtech.org/
fun-proxy speech add --subdomain grace --service http://speech

sudo vi /v2-proxy/speech/traefik/dynamic/middleware-grace-root.toml
	#[http.middlewares.removeprefix-grace-root.addPrefix]
	#prefix = "/GRACE"

#http://speech:8421 <=> https://graceASR.speech.vicomtech.org
fun-proxy speech add --subdomain graceASR --service http://speech:8421

#ASR
curl \
-X POST \
--data "@TESTING/spanish_audio2.request" \
https://graceASR.speech.vicomtech.org/json/
