import numpy as np
from matplotlib import pyplot as plt
import pikepdf



fname = "C:/Users/aiden/Downloads/New folder/test.pdf"
pdf = pikepdf.open(fname)
pdf.save(fname[:-4]+"_alt.pdf", compress_streams=True, linearize=True)