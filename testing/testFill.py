from PIL import Image
from backend.ImageManager import ShowImage, LabelToRGB
from backend.PostProcessing import FillHoles
import numpy as np
import skimage.measure

testImage = skimage.measure.label(np.array(Image.open('testImage.png').convert(mode="L")))


ShowImage(LabelToRGB(testImage))
ShowImage(LabelToRGB(FillHoles(testImage)))