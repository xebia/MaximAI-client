sudo apt update
sudo apt install git
git clone https://github.com/seeed-studio-projects/seeed-voicecard.git
cd seeed-voicecard
sudo ./install.sh
sudo reboot now

# List record and play devices
#aplay -l
#arecord -l