import logging

# from telegram.ext import Updater, CommandHandler, MessageHandler, filters

from telegram.ext import Updater, CommandHandler,  MessageHandler,Filters
from fastai.vision.all import load_learner

import numpy as np
import moviepy.editor as mp
import librosa
import matplotlib.pyplot as plt
import librosa.display
from PIL import ImageMath
from fastai.data.external import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# UNCOMMENT the following if running the bot LOCALLY ON WINDOWS
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

def start(update, context):
    update.message.reply_text(
        "Bot by @avijit on Twitter \n\n"
        "Just send me a short (min 2s) video of your car engine running with the hood up and I'll try to tell you if it is running normally or possibly knocking.\nI will only look at 2 seconds in the middle of the video.\nYour video will not be saved. An example of what I'm expecting can be seen here https://youtu.be/qBAbQakgK60"
    )


def help_command(update, context):
    update.message.reply_text(
        'I will tell you if your car engine is running normally or knocking. Send a short video.')


def load_model():
    global model
    model = load_learner('Models\model_v2.pkl')
    print('Model loaded')


def create_spectrogram(filename):
    clip = mp.VideoFileClip(filename)
    clip_start = (clip.duration/2)-1
    clip_end = (clip.duration/2)+1
    clip = clip.subclip(clip_start, clip_end)
    sr = clip.audio.fps
    y = clip.audio.to_soundarray()
    y = y[..., 0:1:1].flatten()

    D = librosa.stft(y)
    D_harmonic, D_percussive = librosa.decompose.hpss(D)
    rp = np.max(np.abs(D))

    side_px = 256
    dpi = 150
    plot = plt.figure(figsize=(side_px/dpi, side_px/dpi))

    CQT = librosa.amplitude_to_db(
        np.abs(librosa.cqt(librosa.istft(D_percussive), sr=sr)), ref=np.max)
    p = librosa.display.specshow(CQT, x_axis=None, y_axis=None)
    plt.axis('off')

    plot.canvas.draw()

    im_data = np.fromstring(plot.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    im_data = im_data.reshape(plot.canvas.get_width_height()[::-1] + (3,))
    return im_data


def infer_knocking(update, context):
    user = update.message.from_user
    video_file = update.message.video.get_file()
    video_file.download('user_video.mp4')
    label = model.predict(create_spectrogram('user_video.mp4'))[0]
    if label == "normal":
        update.message.reply_text(
            "Your engine seems to be running well. If you are suspecting problems with your car, please contact a mechanic. I'm just a stupid bot and I'm giving my opinion after seeing literally 15 videos on youtube.")
    else:
        update.message.reply_text(
            "Your engine could be knocking. If you are suspecting problems with your car, please contact a mechanic. I'm just a stupid bot and I'm giving my opinion after seeing literally 15 videos on youtube.")


def main():
    #load_model()
    # async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    #     await update.message.reply_text(f'Hello {update.effective_user.first_name}')


    # app = ApplicationBuilder().token("5801822958:AAEjeSEC7wfK07JrUr-LqPNlOGLbvCnNVG8").build()

    # app.add_handler(CommandHandler("hello", hello))
    # app.add_handler(MessageHandler(filters.video, infer_knocking))

    # app.run_polling()
    load_model()
    updater = Updater(token="5801822958:AAEjeSEC7wfK07JrUr-LqPNlOGLbvCnNVG8",use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))

    dp.add_handler(MessageHandler(Filters.video, infer_knocking))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
