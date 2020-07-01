from retrieval_wrapper import RetrievalWrapper
from models import mutopia_ccal_cont as model
from utils.data_pools import prepare_piece_data, AudioScoreRetrievalPool, NO_AUGMENT
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
# path to MSMD data set
DATA_ROOT_MSMD = '/Users/abdelrahman/Desktop/Bachelor References/msmd_aug'

# for now we select just some pieces
# (this could be also the entire training set)
piece_names = ['BachCPE__cpe-bach-rondo__cpe-bach-rondo', 'BachJS__BWV259__bwv-259' , 'AdamA__giselle__giselle']

all_piece_images = []
all_piece_specs = []
all_piece_o2c_maps = []
for piece_name in piece_names:

    piece_image, piece_specs, piece_o2c_maps = prepare_piece_data(DATA_ROOT_MSMD, piece_name, require_audio=False)

    # keep stuff
    all_piece_images.append(piece_image)
    all_piece_specs.append(piece_specs)
    all_piece_o2c_maps.append(piece_o2c_maps)

# path to network parameters
param_file = "/Users/abdelrahman/Desktop/resultdumper/mutopia_ccal_cont_est_UV/params_all_split_mutopia_full_aug.pkl"

# this function is called before a snippet is fed to the network
# (for the present model it resizes the image by a factor of 2)
prepare_sheet_img = model.prepare

# initialize retrieval wrapper
embed_network = RetrievalWrapper(model, param_file, prepare_view_1=prepare_sheet_img, prepare_view_2=None)

# this are dimensions of sheet image snippet and audio excerpt
snippet_shape = model.INPUT_SHAPE_1[1:]
excerpt_shape = model.INPUT_SHAPE_2[1:]
print("snippet_shape", snippet_shape)
print("excerpt_shape", excerpt_shape)

pool = AudioScoreRetrievalPool(all_piece_images, all_piece_specs, all_piece_o2c_maps, data_augmentation=NO_AUGMENT)
#Here we initialize the candidate image tuples and query spectogram

cSheets = np.zeros((10, 1, 160, 200) , dtype=np.float32)
qSpec = np.zeros((1, 1, 92, 42) , dtype=np.float32)
print(cSheets)

#Here we fill the candidate sheets initially zero array with our snippet image data as well as the query spectogram

for i in range(10):
    sheet, spec = pool[i]
    cSheets[i] = sheet
    qSpec = spec


#print(cSheets[0])
#plt.figure()
#plt.clf()
#plt.imshow(qSpec[0 , 0] , cmap='viridis' , origin='lower')
#plt.show()

sheet_codes = embed_network.compute_view_1(cSheets)
print("Sheet codes shape" , sheet_codes.shape)

spec_codes = embed_network.compute_view_2(qSpec)
print("Audio codes shape" , spec_codes.shape)

dist = pairwise_distances(sheet_codes , spec_codes)
min = dist[0]
index = 0
for i in range(len(dist)):
    if min > dist[i]:
        min = dist[i]
        index = i

print(index)
print(dist)
plt.figure()
plt.clf()
plt.imshow(cSheets[index , 0] , cmap='gray' , origin='lower')
plt.show()

plt.figure()
plt.clf()
plt.imshow(qSpec[0,0] , cmap='viridis' , origin='lower')
plt.show()

