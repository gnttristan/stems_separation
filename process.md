1) Download - Download 100 musics from my latest soundcloud likes with yt-dlp library, filter out musics > 6min,
You can look at how yt-dlp works in ../Led player node

2) Preprocessing - Run Demucs on all musics, and on each full/vocals/drums/bass/others run SFTT - 200 bands (mono)
    save one dataframe per stem (5 in total with full), and the frequencies/amplitudes array for each music (100 columns)

3) Annotation - From per-frequencies amplitudes stems values, set 1 if not all amplitudes == 0 else 0, this will allow 
    to know if the stem is currently active on the music (vocal detected / not detected)
    Each music will have a annotation nparray of shape (length, 4) with 0/1, values

4) Features extraction - for full music and for 20 frequencies splits (10 frequencies by split), we will get for each frame the following features :
-    Maximum amplitude               
-    Largest magnitude per frame
-    Nonzero count                   
-    Spectral centroid               
-    Amplitude standard deviation    
-    Multiband entropy               
-    Multiband onset                 
-    Multiband spectral centroid     
-    Multiband spectral flux         
-    Multiband sign-change rate      
-    Multiband mid/side energy       
-    Multiband crest factor

It will give for each frame 11 * 10 = 110 features values
Each features values will be normalized between -1 and 1
The feature logic is already made in ../stems_recognition, take a look and do the exact same thing (split_size also)

5) Model - train a model that will predict from these values 4 outputs (0 or 1) that represent if each stem is present, then save it
6) Inference - load the model and run it on a new music