import sys
import os
from os.path import join as osjoin
from os.path import isdir as isdir
from os.path import isfile as isfile
from os.path import dirname as dirname
from shutil import copyfile
import time
import numpy as np
from cv2 import cv2
import PIL.Image as Image
from threading import Thread

def ensure_isdir(path):
    """Ensure the path exists

    Args:
        path (str): target path
    """
    if not isdir(path):
        os.mkdir(path)

def emphasis_print(sentence):
    print("#################################################")
    print("#")
    print("# ", sentence)
    print("#")
    print("#################################################")

class PreProc(object):
    def __init__(self, datatype, materials):
        """Create an instance of PreProc based on materials and data type

        Args:
            materials (Dict): Materials names as keys and densities as values.
            datatype (str): "nodal" or "boundary"
        """
        if datatype not in ["nodal", "boundary"]:
            print("Datatype %s not recognised. Only \"nodal\" and \"boundary\" supported. " % datatype)
            raise NotImplementedError
        self.materials = materials
        self.cur_dir = dirname(__file__)
        self.data_dir = osjoin(self.cur_dir, dirname('../../data/'))
        self.temp_dir = osjoin(self.data_dir, "temp")
        self.num_res_dir = osjoin(self.data_dir, "NumericalResults")
        self.shockdataset_dir = osjoin(self.data_dir, "shock-datasets")
    
    def unify_height(self, np_img, width, ratio=2): 
        """Unify the height to avoid remaining decimal problems.

        Args:
            np_img (np.ndarray): input image to be unified
            width (int): width of the image
            ratio (int, optional): ratio of width to height. Defaults to 2.

        Returns:
            unified (bool): defines if the image has been unified
            Image.fromarray(np_img.astype('uint8')) (PIL.PNG): the output unified image
        """
        shape = np_img.shape
        h = shape[0]
        true_h = int(width / ratio)
        unified = False
        if h - true_h > 0:
            np_img = np_img[:true_h]
            unified = True
        if h - true_h < 0:
            diff_rows = int(true_h - h)
            fill_in = np.tile(np_img[-1], (diff_rows, 1, 1))
            np_img = np.concatenate((np_img, fill_in))
            unified = True
        return unified, Image.fromarray(np_img.astype('uint8'))

    def clean_numerical_results_to_tmp_dir(self, data_type, material, scale=5, width=640, zoom=1):
        """Read EPS files from NumericalResults and crop the target domain, 
            and then save the domain as according to the width after unified with height.

            Once for SINGLE material. Will be done in multithreading.

        Args:
            data_type (str): either "nodal" or "boundary
            material (str): material name
            file_form (str, optional): save file format. Defaults to "png".
            scale (int, optional): Select a scale for reading .eps file from NumericalResults. Defaults to 5.
            width (int, optional): Set the width of the image to be output. Defaults to 640.
            zoom (int, optional): Select a scale to read from . Defaults to 1.
        """
        if data_type != "nodal" and data_type != "boundary":
            raise NotImplementedError
        
        mat_dir_name = data_type + material                                                 # e.g. nodalBasalt
        src_path = osjoin(self.num_res_dir, mat_dir_name)   # e.g. ../data/NumericalResults/nodalBasalt/
        if not isdir(src_path):
            print("No src path %s found" % src_path)
            return
            
        # boundaryTemp or nodalTemp    
        data_type += "Temp"                                                                          
        des_path = osjoin(self.temp_dir, data_type)                                                      
        results_png_des_path = osjoin(des_path, "ResultsPNG")                                           
        results_eps_des_path = osjoin(des_path, "ResultsEPS")                                           
        png_des_path = osjoin(results_png_des_path, mat_dir_name)                           
        eps_des_path = osjoin(results_eps_des_path, mat_dir_name)                           
        
        ensure_isdir(self.data_dir)                     # e.g. ../data/
        ensure_isdir(self.temp_dir)                     # e.g. ../data/temp/    
        ensure_isdir(des_path)                          # e.g. ../data/temp/nodalTemp/             
        ensure_isdir(results_png_des_path)              # e.g. ../data/temp/nodalTemp/ResultsPNG/   
        ensure_isdir(results_eps_des_path)              # e.g. ../data/temp/nodalTemp/ResultsEPS/   
        ensure_isdir(png_des_path)                      # e.g. ../data/temp/nodalTemp/ResultsPNG/Basalt/*.png/
        ensure_isdir(eps_des_path)                      # e.g. ../data/temp/nodalTemp/ResultsEPS/Basalt/*.eps/

        cnt = 0
        unified_cnt = 0
        src_img_names = os.listdir(src_path)
        print("Starting cleaning...")
        start = time.time()
        for img_name in src_img_names:
            src_img_path = osjoin(src_path, img_name)
            png_des_img_path = osjoin(png_des_path, img_name.replace('.eps', '.png'))
            
            # read image from EPS format with scale defined
            raw = Image.open(src_img_path)
            raw.load(scale=scale)
            raw = np.asarray(raw)
            img = raw.copy()

            # clean up the image
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(img, 10, 250)
            (cntrs, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            unified, unified_img = False, None
            for c in cntrs:
                x,y,w,h = cv2.boundingRect(c)
                if w>500 and h>250:
                    cropped = img[y:y+h,x:x+w]
                    converted_img = Image.fromarray(cropped.astype('uint8'))
                    # resize the image
                    wpercent = (width / float(cropped.shape[0]))
                    hsize = int(float(cropped.shape[1]) * float(wpercent))
                    converted_img.thumbnail((width, hsize), Image.ANTIALIAS)

                    np_img = np.asarray(converted_img)
                    unified, unified_img = self.unify_height(np_img, width)

            unified_img.save(png_des_img_path)

            cnt += 1
            unified_cnt += unified
            if (cnt % 500 == 0):
                temp = time.time()
                time_used = temp - start
                time_per = time_used / (cnt + 1)
                approx_rem = (len(src_img_names) - (cnt + 1)) * time_per
                print("%s %s Images Converted (%s images unified); time used %s; approx. \
                    %ss remaining..." % (material, cnt, unified_cnt, time_used, approx_rem))

    def reshape(self, old_img, size):
        """Reshape the image into new size.

        Args:
            old_img (PIL.Image): Image opened by PIL
            size (list): Size of new image to be reshaped into: [width, height]

        Returns:
            PIL.Image: Reshape image as PIL.Image class.
        """
        new_img = old_img.resize(size, Image.ANTIALIAS)
        return new_img

    def reshape_imgs_and_store_to_dataset(self, type_dir, og_shape=[640, 320], new_shapes=[[96, 48], [128, 64]]):
        """[summary]

        Args:
            type_dir (str): "nodal" or "boundary"
            og_shape (list, optional): The original [width, height]. Defaults to [640, 320].
            new_shapes (list, optional): New [width, height]. Defaults to [[96, 48], [128, 64], [256, 128]].
        """
        def name_with_shape(fname, shp):
            return fname + '_' + str(shp[1]) + 'x' + str(shp[0])

        clnd_parent_dir = osjoin(self.data_dir, type_dir + "Temp", "ResultsPNG")        # ../data/*Temp/
        if not isdir(clnd_parent_dir):
            print("No temp directory found %s" % clnd_parent_dir)
            return

        des_path = osjoin(self.shockdataset_dir, type_dir)            # ../data/shock-datasets/nodal/
        ensure_isdir(des_path)

        og_des_path = osjoin(des_path, name_with_shape(type_dir, og_shape))                    # e.g. ../data/shock-datasets/nodal/nodal_320x640/
        ensure_isdir(og_des_path)
        og_des_path = osjoin(og_des_path, "ResultsPNG")                                           # ../data/shock-datasets/nodal/nodal_320x640/ResultsPNG/
        ensure_isdir(og_des_path)
        reshape_des_paths = []
        if new_shapes:
            for shp in new_shapes:
                reshape_des_path = osjoin(des_path, name_with_shape(type_dir, shp))            # e.g. ../data/shock-datasets/nodal_48x96/
                ensure_isdir(reshape_des_path)
                reshape_des_path = osjoin(reshape_des_path, "ResultsPNG")                      # ../data/shock-datasets/nodal_48x96/ResultsPNG/
                ensure_isdir(reshape_des_path)
                reshape_des_paths.append(reshape_des_path)

        cnt = 1
        for mat in os.listdir(clnd_parent_dir):
            data_mat_path = osjoin(clnd_parent_dir, mat)
            for old_name in os.listdir(data_mat_path):
                mat_name = mat.replace(type_dir, "")
                name_list = old_name.split('_')
                if name_list[0] != mat_name:
                    name_list[0] = mat_name+ '_' + str(self.materials[mat_name])
                new_name = '_'.join(name_list)
                old_i_path = osjoin(data_mat_path, old_name)
    
                new_i_path = osjoin(data_mat_path, new_name)
                if old_name != new_name:
                    os.rename(old_i_path, new_i_path)

                copyfile(new_i_path, osjoin(og_des_path, new_name))
                if new_shapes:
                    for i, shp in enumerate(new_shapes):
                        img = Image.open(osjoin(og_des_path, new_name))
                        reshaped_img = self.reshape(img, shp)
                        reshaped_img.save(osjoin(reshape_des_paths[i], new_name))
                if cnt % 500 == 0:
                    print("%s Renamed %s files" % (mat, cnt))
                cnt += 1


    #############################################################
    #                                                           #
    # Generating corresponding initial conditions from          #
    # ../data/shock-dataset/nodal/nodal_320x640/ResultsPNG      #
    #                                                           #
    #############################################################

    def setup_input(self, img, density, force, seps=None, scale=[0.085, 127], pad=10):
        """Set up the initial condition image for corresponding numerical results

        Args:
            img (PIL.Image): target numerical result image
            density (float): density of the material
            force (float): force casted on the upper boundary
            seps (list, optional): for nodal force initialisation. Defaults to None.
            scale (list, optional): the scale to transform value as colours. Defaults to [0.085, 127].
            pad (int, optional): a padding for the initial condition image. Defaults to 10.

        Returns:
            [type]: [description]
        """
        w, h = img.size
        in_img = np.ones((h, w, 3)).astype('int32')
        in_img[:, :, -1] = in_img[:, :, -1] * 200   # set background colour as blue
        
        # set background via density
        density_scale = int(density * scale[0])
        in_img[pad:-pad, 2*pad:-2*pad, :] = in_img[pad:-pad, 2*pad:-2*pad, :]*0 \
                                            + density_scale # grey scale as density

        # set support boundary
        in_img[-pad:, 2*pad:-2*pad, -1] = in_img[-pad:, 2*pad:-2*pad, -1]*0       # set b channel to 0
        in_img[-pad:, 2*pad:-2*pad, 0] = in_img[-pad:, 2*pad:-2*pad, 0]*0 + 255 # redli

        # set force
        force_scale = int(force * scale[1])
        if seps is None:
            in_img[:pad, 2*pad:-2*pad, -1] = in_img[:pad, 2*pad:-2*pad, -1]*0                 # r & g
            in_img[:pad, 2*pad:-2*pad, :-1] = in_img[:pad, 2*pad:-2*pad, :-1]*0 + force_scale # r & g
        else:
            pure_w = w - 2 * 2 * pad
            nodenum = seps[-1] - seps[0]
            _, m, r = np.diff(seps) / nodenum
            m_left = max(pad + int(pure_w * r) + 1, 2*pad)
            m_right = min(m_left + int(pure_w * m), w - 2*pad)
            in_img[:pad, m_left:m_right, -1]= in_img[:pad, m_left:m_right, -1]*0
            in_img[:pad, m_left:m_right, :-1]= in_img[:pad, m_left:m_right, :-1]*0 + force_scale # r & g

        return in_img

    def generate_input(self, type_dir, scale=[0.085, 127]):
        """Generate inputs i.e. initial conditons via 
            numerical results and saved information

        Args:
            src_dir (str): [description]
            nodal (bool, optional): [description]. Defaults to False.
            pad (int, optional): [description]. Defaults to 10.
            scale (list, optional): [description]. Defaults to [0.085, 127].
        """
        parent_dir = osjoin(self.shockdataset_dir, type_dir)
        if not isdir(parent_dir):
            print("shock-datasets sub directory not found")
            return

        src_paths, des_paths, pads = [], [], [1,2,3,4,5]
        for i in os.listdir(parent_dir):
            pads.append(int(i.split('_')[1].split('x')[0]) // 10)
            if not i:
                print("No sub directory found in %s" % parent_dir)
                return
            src_path = osjoin(parent_dir, i, "ResultsPNG")
            if not isdir(src_path):
                print("%s not found as a directory" % src_path)
                continue
            des_path = osjoin(parent_dir, i, "InitialCondition")
            ensure_isdir(des_path)
            src_paths.append(src_path)
            des_paths.append(des_path)

        for i in range(len(src_paths)):
            pad = pads[i]
            src_img_names = os.listdir(src_paths[i])
            src_path = src_paths[i]
            des_path = des_paths[i]
            print("Starting to generate data for %s" % src_path)
            cnt = 0
            start = time.time()
            for pname in src_img_names:
                # get parameters in filename
                img_name_without_suffix = pname[:-4]      # get the name without .png
                paras = img_name_without_suffix.split('_')
                to_float = lambda x : float(x.replace('d', '.'))
                density = int(paras[1])
                force = to_float(paras[4])                          # in N
                
                seps = None
                if type_dir == "nodal":
                    seps = np.array([int(i) for i in paras[2].split('-')])

                img_path = osjoin(src_path, pname)
                img = Image.open(img_path)
                in_img = self.setup_input(img, density, force, seps=seps, pad=pad, scale=scale)
                img = Image.fromarray(in_img.astype('uint8'))
                input_des_path = osjoin(des_path, pname)
                img.save(input_des_path)
                cnt += 1
                if (cnt % 500 == 0):
                    temp = time.time()
                    time_used = temp - start
                    time_per = time_used / (cnt + 1)
                    approx_rem = (len(src_img_names) - (cnt + 1)) * time_per
                    print("%s %sth Input Image Created; time used %s; approx. %ss remaining..." % (paras[0], cnt, time_used, approx_rem))

def main_preprocessing(datatype, materials):

    prep = PreProc(datatype, materials)

    ###########################################
    emphasis_print("Starting cleaning numerical results and storing to Temp directory")
    n = len(materials)
    threads = []
    for i in range(n):
        t = Thread(target=prep.clean_numerical_results_to_tmp_dir, args=(datatype, list(materials)[i], ))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()

    ###########################################
    emphasis_print("Starting renaming, storing and reshaping dataset in Temp directory")
    prep.reshape_imgs_and_store_to_dataset(datatype)


    ###########################################
    emphasis_print("Starting generating initial condition images")
    prep.generate_input(datatype)

if __name__ == "__main__":
    # python3 preprocessing.py -h
    # python3 preprocessing.py -d nodal -m Dolomite Limestone Sandstones Basalt Granite Chalks
    import argparse, getopt
    description = """\
                    Preprocessing to crop the raw output from ANSYS where only the stress field was saved. \n\
                    The cropped image will then be renamed, copied and reshaped. \n\
                    Inputs corresponding to the output images will then be generated. \n\n\
                    Material dictionary includes: \n\
                        Limestone, Dolomite, Chalks, Sandstones, Granite and Basalt.\n\
                    Examples: \n\
                        python3 preprocessing.py -d nodal -m Dolomite Limestone Sandstones Basalt Granite Chalks
                    """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--datatype", "-d", 
                        help="for \"nodal\" or \"boundary\" types of datasets")
    parser.add_argument("--materials", "-m", 
                        nargs='*',
                        help="to assign materials")
    args = parser.parse_args()
    datatype = args.datatype
    mat_dic = {"Limestone":2160, "Dolomite":2840, "Chalks":2499, "Sandstones":2600, "Granite":2750, "Basalt":3000}
    cur_mat = {}
    for mat in args.materials:
        if mat not in mat_dic:
            print("%s not found in dictionary. Only Limestone \
                   Dolomite, Chalks, Sandstones, Granite and Basalt are supported.")
            raise NotImplementedError
        cur_mat.update({mat: mat_dic[mat]})
    print(cur_mat)

    main_preprocessing(datatype, cur_mat)



    