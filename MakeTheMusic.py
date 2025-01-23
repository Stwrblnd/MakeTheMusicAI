import tensorflow as tf
import pickle
from tkinter import *
from tkinter import filedialog
import os
import random
from pydub import AudioSegment
from pychord import Chord
from midiutil import MIDIFile
import tempfile
import numpy as np

def load_model_and_dictionaries(model_name):

    model = tf.keras.models.load_model(f'{model_name}.h5')
    
    with open(f'{model_name}_chord_to_int.pkl', 'rb') as f:
        chord_to_int = pickle.load(f)
    with open(f'{model_name}_int_to_chord.pkl', 'rb') as f:
        int_to_chord = pickle.load(f)
    
    return model, chord_to_int, int_to_chord

options_without_any = ['C', 'Db', 'D', 'Eb', 'E', 
                       'F', 'Gb', 'G', 'Ab', 'A', 
                       'Bb', 'B', 'Cm', 'Dbm', 'Dm',
                       'Ebm', 'Em', 'Fm', 'Gbm', 'Gm', 
                       'Abm', 'Am', 'Bbm', 'Bm']

def generate_chords(mood, start_chord, num_chords=3):
    
    model_name = f"{mood}_model"
    model, chord_to_int, int_to_chord = load_model_and_dictionaries(model_name)
    
    if start_chord not in chord_to_int:
        start_chord = random.choice(options_without_any)
    
    sequence = [chord_to_int[start_chord]]
    generated_chords = [start_chord]  

    for _ in range(num_chords):
        x_pred = np.array(sequence[-3:])[np.newaxis, :]
        x_pred = np.pad(x_pred, ((0, 0), (max(0, 3-x_pred.shape[1]), 0)), mode='constant', constant_values=0)
        prediction = model.predict(x_pred, verbose=0)[0]
        
        next_index = np.random.choice(range(len(prediction)), p=prediction)
        generated_chords.append(int_to_chord[next_index])
        sequence.append(next_index)

    return generated_chords

def chords_to_notes(chord_progression):
    notes_of_chord_progression = []
    for chord in chord_progression:
        c = Chord(chord)
        notes_of_chord_progression.append(c.components())
    return notes_of_chord_progression

def notes_to_midi_with_bass(chord_notes):
    note_mapping = {
        "C": 60, "Db": 61, "C#": 61, "D": 62, "Eb": 63, "D#": 63, "E": 64,
        "F": 65, "Gb": 66, "F#": 66, "G": 67, "Ab": 68, "G#": 68, "A": 69,
        "Bb": 70, "A#": 70, "B": 71, "Cb": 71, "Fb": 64
    }

    midi_chords = []
    bass_notes = []
    for chord in chord_notes:
        midi_chord = [note_mapping[note] for note in chord]
        midi_chords.append(midi_chord)
        bass_notes.append(note_mapping[chord[0]] - 12) 
    return midi_chords, bass_notes

def generate_midi_with_bass(notes_of_chord_progression, output_file, repeats, tempo, add_bass):
    chord_notes, bass_notes = notes_to_midi_with_bass(notes_of_chord_progression)
    midi = MIDIFile(2 if add_bass else 1)  
    track_chords = 0
    track_bass = 1 if add_bass else None 
    channel = 0
    volume = 100
    time = 0
    midi.addTempo(track_chords, time, tempo)
    if track_bass is not None:
        midi.addTempo(track_bass, time, tempo)
    
    for z in range(0, repeats * 16, 16):
        for i, chord in enumerate(chord_notes):
            for pitch in chord:
                midi.addNote(track_chords, channel, pitch, i * 4 + z, 4, volume)

    if track_bass is not None:
        for z in range(0, repeats * 16, 16):
            for i, bass in enumerate(bass_notes):
                midi.addNote(track_bass, channel, bass, i * 4 + z, 4, volume)
    
    with open(output_file, "wb") as output_file:
        midi.writeFile(output_file)

def add_drums_to_audio(audio, drum_style, tempo, drum_volume=0.5):
    if drum_style == "No":
        return audio

    if drum_style == "Rock":
        kick = AudioSegment.from_file("sounds/rock_kick.wav")
        snare = AudioSegment.from_file("sounds/rock_snare.wav")
        hi_hat = AudioSegment.from_file("sounds/rock_hihat.wav")
    elif drum_style == "Electronic":
        kick = AudioSegment.from_file("sounds/electronic_kick.wav")
        snare = AudioSegment.from_file("sounds/electronic_snare.wav")
        hi_hat = AudioSegment.from_file("sounds/electronic_hihat.wav")


    kick = kick - (1 - drum_volume) * 30  
    snare = snare - (1 - drum_volume) * 30
    hi_hat = hi_hat - (1 - drum_volume) * 30

    beat_duration = 60000 / tempo
    eighth_duration = beat_duration / 2

    one_bar = AudioSegment.silent(duration=4 * beat_duration) 

    one_bar = (
        one_bar.overlay(kick, position=0)  
        .overlay(snare, position=beat_duration)  
        .overlay(kick, position=2 * beat_duration)  
        .overlay(snare, position=3 * beat_duration) 
    )

    for i in range(0, 8):
        one_bar = one_bar.overlay(hi_hat, position=i * eighth_duration)

    drum_track = one_bar * (len(audio) // len(one_bar) + 1)

    drum_track = drum_track[:len(audio)]

    mixed_audio = audio.overlay(drum_track)
    return mixed_audio

def midi_to_mp3_with_drums(midi_file, soundfont, mp3_file, drum_style, tempo):
    wav_file = mp3_file.replace('.mp3', '.wav')
    os.system(f'fluidsynth -ni {soundfont} {midi_file} -F {wav_file} -r 44100')
    audio = AudioSegment.from_wav(wav_file)
    audio_with_drums = add_drums_to_audio(audio, drum_style, tempo)
    audio_with_drums.export(mp3_file, format='mp3')

    os.remove(wav_file)

def soundfont_choose(sf_name):
    if sf_name == 'Piano':
        sf = 'sounds/GeneralUser_GS_v1.471.sf2'
    elif sf_name == 'Marimba':
        sf = 'sounds/marimba-deadstroke.sf2'
    elif sf_name == 'Old video games':
        sf = 'sounds/PICO-8_1.1.2.sf2'
    return sf

last_generated_chords = []
root = Tk()
root.title("MakeTheMusic")
root.geometry("700x700")

header = LabelFrame(root, text="Sound", padx=10, pady=10)
header.pack(fill="x", padx=10, pady=10)

input_wrap = Frame(header)

options_major_minor = ["happy", "sad"]
value_inside = StringVar(root)
value_inside.set(options_major_minor[0])
dropdown_minor_major = OptionMenu(input_wrap, value_inside, *options_major_minor)
dropdown_minor_major.pack(side=LEFT)

options_chord = ['Any', 'C', 'Db', 'D', 'Eb', 'E', 'F',
                 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'Cm', 
                 'Dbm', 'Dm', 'Ebm', 'Em', 'Fm', 'Gbm', 'Gm', 
                 'Abm', 'Am', 'Bbm', 'Bm']
value_inside_n = StringVar(root)
value_inside_n.set(options_chord[0])
dropdown_chord = OptionMenu(input_wrap, value_inside_n, *options_chord)
dropdown_chord.pack(side=LEFT)

input_wrap.pack()

label_wrapper = Frame(header)
l_root = Label(label_wrapper, text="Mood").pack(side=LEFT)
l_prob = Label(label_wrapper, text="First chord").pack(side=LEFT)
label_wrapper.pack()

res_field_text = StringVar()
res_field = Label(root, textvariable=res_field_text)

msg_text = StringVar()
msg_field = Label(root, textvariable=msg_text, fg="white")

def on_generate():
    global last_generated_chords
    mood = value_inside.get()
    selected_start_chord = value_inside_n.get()
    
    if selected_start_chord == 'Any':
        selected_start_chord = random.choice(options_without_any)
    
    new_chords = generate_chords(mood, selected_start_chord)
    
    while new_chords == last_generated_chords:
        new_chords = generate_chords(mood, selected_start_chord)
    
    last_generated_chords = new_chords
    
    res_field_text.set(' '.join(new_chords))
    msg_text.set("Chords have been generated!")

    settings_label.pack()
    tempo_label.pack()
    tempo_entry.pack()
    repetitions_label.pack()
    repetitions_entry.pack()
    synth_label.pack()
    synth_options.pack()
    generate_midi_btn.pack()
    generate_mp3_btn.pack()


def on_generate_midi():
    tempo = int(tempo_entry.get())
    repetitions = int(repetitions_entry.get())
    notes = chords_to_notes(last_generated_chords)
    output_file = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI files", "*.mid")])  
    if not output_file:
        return  
    
    add_bass = bass_line_var.get()
    generate_midi_with_bass(notes, output_file, repetitions, tempo, add_bass)
    msg_text.set("MIDI file has been generated" + (" with the bassline" if add_bass else ""))
 

def on_generate_mp3():
    if synth_var.get() == "Other":
        soundfont = synth_var_custom.get()
    else:
        soundfont = soundfont_choose(synth_var.get())

    repetitions = int(repetitions_entry.get())
    tempo = int(tempo_entry.get())
    drum_style = drum_style_var.get()
    
    chords_notes = chords_to_notes(last_generated_chords)

    add_bass = bass_line_var.get() 

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_midi:
        generate_midi_with_bass(chords_notes, temp_midi.name, repetitions, tempo, add_bass)

        mp3_file = filedialog.asksaveasfilename(defaultextension=".mp3", filetypes=[("MP3 files", "*.mp3")])
        if not mp3_file:
            return

        midi_to_mp3_with_drums(temp_midi.name, soundfont, mp3_file, drum_style, tempo)
        msg_text.set("MP3 has been generated" + (" with the bassline" if add_bass else ""))

    os.remove(temp_midi.name)

space_frame = Frame(root)
space_frame.pack(pady=5) 

gen_btn = Button(root, text="Generate a chord progression", width=50, command=on_generate)
gen_btn.pack()

res_field.pack(padx=50, pady=5)
msg_field.pack(padx=50, pady=5)

separator_label = Label(root, text="_______________________________________")
separator_label.pack()

settings_label = Label(root, text="Output settings:")

tempo_label = Label(root, text="BPM:")
tempo_entry = Entry(root)
tempo_entry.insert(0, "120") 

repetitions_label = Label(root, text="Repetitions:")
repetitions_entry = Entry(root)
repetitions_entry.insert(0, "1")  

synth_var_custom = StringVar(root)
synth_var_custom.set("")

def on_synthesizer_selected(selected_option):
    if selected_option == "Other":
        filename = filedialog.askopenfilename(filetypes=[("SoundFont files", "*.sf2")])
        if filename:
            synth_var_custom.set(filename)
    else:
        synth_var_custom.set("") 

synth_var = StringVar(root)
synth_var.set("Piano")
synth_label = Label(root, text="Choose the synthesizer for MP3")
synth_options = OptionMenu(root, synth_var, "Piano", "Old video games", "Marimba", "Other", command=on_synthesizer_selected)

space_frame = Frame(root)
space_frame.pack(pady=10) 

drum_style_label = Label(root, text="Add drums:")
drum_style_label.pack()

drum_style_var = StringVar(root)
drum_style_var.set("No") 
drum_style_options = OptionMenu(root, drum_style_var, "No", "Rock", "Electronic")
drum_style_options.pack()

bass_line_var = BooleanVar()
bass_line_checkbox = Checkbutton(root, text="Add the bass line", variable=bass_line_var)
bass_line_checkbox.pack()

generate_midi_btn = Button(root, text="Generate MIDI", command=on_generate_midi)
generate_mp3_btn = Button(root, text="Generate MP3", command=on_generate_mp3)

def open_window_with_text(file_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, file_path)
    with open(full_path, 'r', encoding='utf-8') as file:
        text = file.read()
    new_window = Toplevel(root)
    new_window.title("New window")
    label = Label(new_window, text=text)
    label.pack()


tutorial_button = Button(root, text="Tutorial", command=lambda: open_window_with_text("textfiles/tutorial.txt"))
tutorial_button.pack(side="bottom", anchor="se", padx=10, pady=10)
atribution_button = Button(root, text="Atribution", command=lambda: open_window_with_text("textfiles/atribution.txt"))
atribution_button.pack(side="bottom", anchor="se", padx=10)

root.mainloop()
