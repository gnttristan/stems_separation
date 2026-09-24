from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import streamlit as st
from inference import ROOT, predict, player_html


def main():
    source, output = Path(__file__).parent / 'input', Path(__file__).parent / 'output'
    source.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    st.title('Stem activity player')
    upload = st.file_uploader('Drop or choose music', type=['mp3', 'wav', 'flac', 'ogg', 'm4a'],
                              on_change=lambda: st.session_state.pop('player', None))
    if st.button('Analyze audio', disabled=upload is None):
        st.session_state.pop('player', None)
        try:
            with st.spinner('Detecting vocals, drums, bass and other sounds…'):
                track = source / Path(upload.name).name
                track.write_bytes(upload.getvalue())
                result = predict(track, ROOT / '05_model/output/model.npz')
                np.savez_compressed(output / f'{track.stem}.npz', **result)
                st.session_state.player = player_html(track, result)
        except Exception as error:
            st.error(f'Could not analyze this audio: {error}')
    if 'player' in st.session_state:
        st.iframe(st.session_state.player, height=370)


if __name__ == '__main__':
    main()
