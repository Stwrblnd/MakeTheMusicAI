import tensorflow as tf
import pickle
from tkinter import *
from tkinter import ttk, filedialog, messagebox
import os
import random
from pydub import AudioSegment
from pychord import Chord
from midiutil import MIDIFile
import tempfile
import numpy as np
import pygame

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

def generate_lead_melody(chords_notes, tempo, repetitions):
    if not chords_notes:
        return []

    lead_notes = []
    time_offset = 0  

    for _ in range(repetitions):  
        for chord in chords_notes:
            if len(chord) < 3:
                continue 

            midi_notes = sorted(chord)  
            midi_notes = [note + 12 for note in midi_notes]  

            rhythm_pattern = [1, 1, 2]

            note_time = time_offset 
            for note, duration in zip(midi_notes, rhythm_pattern):
                lead_notes.append((note, note_time, duration))
                note_time += duration  

            time_offset += 4  

    return lead_notes

def generate_midi_with_bass(notes_of_chord_progression, output_file, repeats, tempo, add_bass, add_lead):
    chord_notes, bass_notes = notes_to_midi_with_bass(notes_of_chord_progression)
    
    if add_lead:
        lead_melody_per_chord = [generate_lead_melody([ch], tempo, 1) for ch in chord_notes]
    else:
        lead_melody_per_chord = [] 
    
    midi = MIDIFile(3 if add_lead else (2 if add_bass else 1)) 

    track_chords = 0
    track_bass = 1 if add_bass else None
    track_lead = 2 if add_lead else None
    channel = 0
    volume = 100
    lead_volume = 110  

    time = 0

    midi.addTempo(track_chords, time, tempo)
    if track_bass is not None:
        midi.addTempo(track_bass, time, tempo)
    if track_lead is not None:
        midi.addTempo(track_lead, time, tempo)

    for z in range(0, repeats * 16, 16):
        for i, chord in enumerate(chord_notes):
            for pitch in chord:
                midi.addNote(track_chords, channel, pitch, i * 4 + z, 4, volume)

    if track_bass is not None:
        for z in range(0, repeats * 16, 16):
            for i, bass in enumerate(bass_notes):
                midi.addNote(track_bass, channel, bass, i * 4 + z, 4, volume)

    if track_lead is not None:
        for repeat_index in range(repeats):  
            time_offset = repeat_index * 16 

            for chord_index, chord_melody in enumerate(lead_melody_per_chord):
                chord_start_time = time_offset + chord_index * 4  

                for lead_note, relative_time, duration in chord_melody:
                    absolute_time = chord_start_time + relative_time
                    midi.addNote(track_lead, channel, lead_note, absolute_time, duration, lead_volume)

    with open(output_file, "wb") as output_f:
        midi.writeFile(output_f)

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

def play_mp3():
    generate_mp3()
    if last_generated_mp3 and os.path.exists(last_generated_mp3):
        pygame.mixer.music.load(last_generated_mp3)
        pygame.mixer.music.play()
    else:
        messagebox.showerror("Error", "No MP3 file available to play.")

def stop_mp3():
    pygame.mixer.music.stop()

pygame.mixer.init()
last_generated_chords = []
root = Tk()
root.title("MakeTheMusic")
root.geometry("700x700")

bg_image = None
bg_label = None

def set_background(image_path):
    global bg_image, bg_label
    if bg_label is not None:
        bg_label.destroy()
    try:
        bg_image = PhotoImage(file=image_path)
        bg_label = Label(root, image=bg_image)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        bg_label.lower(header) 
    except TclError:
        messagebox.showerror("Error", f"Image file not found: {image_path}")

header = LabelFrame(root, text="Sound", padx=10, pady=10)
header.pack(fill="x", padx=10, pady=10)

input_wrap = Frame(header)
options_major_minor = ["happy", "sad"]
value_inside = StringVar(root)
value_inside.set(options_major_minor[0])
dropdown_minor_major = ttk.Combobox(input_wrap, values=options_major_minor, state="readonly", textvariable=value_inside)
dropdown_minor_major.pack(side=LEFT)

options_chord = ['Any', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'Cm', 'Dbm', 'Dm', 'Ebm', 'Em', 'Fm', 'Gbm', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm']
value_inside_n = StringVar(root)
value_inside_n.set(options_chord[0])
dropdown_chord = ttk.Combobox(input_wrap, values=options_chord, state="readonly", textvariable=value_inside_n)
dropdown_chord.pack(side=LEFT)

input_wrap.pack()
label_wrapper = Frame(header)
Label(label_wrapper, text="Mood").pack(side=LEFT)
Label(label_wrapper, text="First chord").pack(side=LEFT)
label_wrapper.pack()

res_field_text = StringVar()
res_field = Label(root, textvariable=res_field_text)

output_settings = LabelFrame(root, text="Output settings:")
output_settings.pack_forget()

Label(output_settings, text="BPM:").grid(row=0, column=0, sticky=W)
tempo_entry = Entry(output_settings)
tempo_entry.insert(0, "120")
tempo_entry.grid(row=0, column=1)

Label(output_settings, text="Repetitions:").grid(row=1, column=0, sticky=W)
repetitions_entry = Entry(output_settings)
repetitions_entry.insert(0, "1")
repetitions_entry.grid(row=1, column=1)

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
Label(output_settings, text="Choose the synthesizer for MP3").grid(row=2, column=0, sticky=W)
synth_options = ttk.Combobox(output_settings, values=["Piano", "Old video games", "Marimba", "Other"], state="readonly", textvariable=synth_var)
synth_options.grid(row=2, column=1)
synth_options.bind("<<ComboboxSelected>>", lambda e: on_synthesizer_selected(synth_var.get()))

drum_style_var = StringVar(root)
drum_style_var.set("No")
Label(output_settings, text="Add drums:").grid(row=3, column=0, sticky=W)
drum_style_options = ttk.Combobox(output_settings, values=["No", "Rock", "Electronic"], state="readonly", textvariable=drum_style_var)
drum_style_options.grid(row=3, column=1)

bass_line_var = BooleanVar()
Checkbutton(output_settings, text="Add the bass line", variable=bass_line_var).grid(row=4, columnspan=2, sticky=W)
lead_melody_var = BooleanVar()
Checkbutton(output_settings, text="Add a lead melody", variable=lead_melody_var).grid(row=5, columnspan=2, sticky=W)

def on_generate():
    global last_generated_chords
    mood = value_inside.get()
    selected_start_chord = value_inside_n.get()
    
    if selected_start_chord == 'Any':
        selected_start_chord = random.choice(options_chord[1:])
    
    new_chords = generate_chords(mood, selected_start_chord)
    
    while new_chords == last_generated_chords:
        new_chords = generate_chords(mood, selected_start_chord)
    
    last_generated_chords = new_chords
    res_field_text.set(' '.join(new_chords))
    output_settings.pack(pady=10)
    generate_mp3()
    play_button.pack(side=LEFT, padx=10)
    stop_button.pack(side=RIGHT, padx=10)

def on_generate_midi():
    tempo = int(tempo_entry.get())
    repetitions = int(repetitions_entry.get())
    notes = chords_to_notes(last_generated_chords)
    output_file = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI files", "*.mid")])  
    if not output_file:
        return  
    add_bass = bass_line_var.get()
    add_lead = lead_melody_var.get()
    generate_midi_with_bass(notes, output_file, repetitions, tempo, add_bass, add_lead)

def generate_mp3():
    global last_generated_mp3

    if synth_var.get() == "Other":
        soundfont = synth_var_custom.get()
    else:
        soundfont = soundfont_choose(synth_var.get())

    repetitions = int(repetitions_entry.get())
    tempo = int(tempo_entry.get())
    drum_style = drum_style_var.get()
    
    chords_notes = chords_to_notes(last_generated_chords)

    add_bass = bass_line_var.get() 
    add_lead = lead_melody_var.get()  

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_midi:
        generate_midi_with_bass(chords_notes, temp_midi.name, repetitions, tempo, add_bass, add_lead)

        temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        last_generated_mp3 = temp_mp3.name

        midi_to_mp3_with_drums(temp_midi.name, soundfont, last_generated_mp3, drum_style, tempo)
        os.remove(temp_midi.name)

def save_mp3_file():
    if last_generated_mp3:
        save_path = filedialog.asksaveasfilename(defaultextension=".mp3", filetypes=[("MP3 files", "*.mp3")])
        if save_path:
            os.rename(last_generated_mp3, save_path)
            last_generated_mp3 = save_path  

set_background("background.png") 

Button(root, text="Generate a chord progression", width=50, command=on_generate).pack()

control_frame = Frame(root)
control_frame.pack()

play_button = Button(control_frame, text="▶ Play", command=play_mp3)
stop_button = Button(control_frame, text="■ Stop", command=stop_mp3)

play_button.pack_forget()
stop_button.pack_forget()

res_field.pack(padx=50, pady=5)
Button(output_settings, text="Save MIDI", command=on_generate_midi).grid(row=6, column=0, pady=10)
Button(output_settings, text="Save MP3", command=save_mp3_file).grid(row=6, column=1, pady=10)

bottom_frame = Frame(root)
bottom_frame.pack(side="bottom", fill="x", pady=10)

def open_window_with_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        new_window = Toplevel(root)
        new_window.title(file_path)
        
        text_widget = Text(new_window, wrap="word", width=60, height=20)
        text_widget.insert("1.0", text)
        text_widget.config(state="disabled") 
        text_widget.pack(padx=10, pady=10, expand=True, fill="both")
        
        scrollbar = Scrollbar(new_window, command=text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        text_widget.config(yscrollcommand=scrollbar.set)
        
    except FileNotFoundError:
        messagebox.showerror("Error", f"File not found: {file_path}")


attribution_button = Button(bottom_frame, text="Attribution", width=15, command=lambda: open_window_with_text("textfiles/atribution.txt"))
attribution_button.pack(side="right", padx=10)

tutorial_button = Button(bottom_frame, text="Tutorial", width=15, command=lambda: open_window_with_text("textfiles/tutorial.txt"))
tutorial_button.pack(side="right", padx=10)

root.mainloop()
