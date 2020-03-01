animation.mp4: sun_path.py
	mkdir .frames -p
	python3 sun_path.py
	cd .frames/ ; ffmpeg -y -r 10 -i frame_%04d.png ../animation.mp4

clean:
	cd .frames/ ; rm frame_*.png
