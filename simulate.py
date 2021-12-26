# The purpose of this python script is to perform color blindness simulation
# mainly protanopic color blindness. I have used this to test the logic for the 
# web app hosted at: https://gifted-carson-21452d.netlify.app/

# A sample input and output image is provided in the repo

#Reference paper https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.496.7153&rep=rep1&type=pdf
import numpy as np
import math
import imageio
import sys
from numpngw import write_png


def remove_gamma_internal(c):
    if c <= 0.04045 * 255:
        return c/(255*12.92)
    return math.pow((((c/255)+0.055)/1.055), 2.4)


def remove_gamma(v):
    return np.asarray([remove_gamma_internal(v[0]), remove_gamma_internal(v[1]), remove_gamma_internal(v[2])])


def convert_rgb_to_lms(v):
    t = np.asarray([[0.31399022, 0.63951294, 0.04649755], [
                   0.15537241, 0.75789446, 0.08670142], [0.01775239, 0.10944209, 0.87256922]])
    return np.dot(t, v.transpose())


def convert_lms_to_rgb(v):
    t = np.asarray([[5.47221206, -4.6419601, 0.16963708], [-1.1252419,
                   2.29317094, -0.1678952], [0.02980165, -0.19318073, 1.16364789]])
    return np.dot(t, v.transpose())


def apply_gamma(v):
    return np.asarray([apply_gamma_internal(v[0]), apply_gamma_internal(v[1]), apply_gamma_internal(v[2])])


def apply_gamma_internal(v):
    if v <= 0.0031308:
        return 255 * (12.92 * v)
    return 255 * (1.055 * math.pow(v, 0.4167)-0.055)


def read_image(name):
    return imageio.imread(name)


def write_image(file_name, img):
    write_png(file_name, img)


def derive_planes(neutral, lms1, lms2):
    p1 = np.cross(lms1, neutral)
    p2 = np.cross(lms2, neutral)

    return p1, p2


def project_on_plane(n, v):
    v[0] = - (n[1]*v[1] + n[2]*v[2]) / n[0]
    return v


def get_protanope_plane_of_projection(p1, p2, neutral, v):
    if v[1] != 0 and v[2]/v[1] < neutral[2]/neutral[1] :
        return p2
    return p1

def simulate_protanopia(p1, p2, neutral, img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            converted_value = convert_rgb_to_lms(remove_gamma(img[i, j]))
            n = get_protanope_plane_of_projection(
                p1, p2, neutral, converted_value)
            converted_value = project_on_plane(n, converted_value)
            converted_value = convert_lms_to_rgb(converted_value)
            converted_value = apply_gamma(converted_value)
            img[i, j] = clamp(converted_value)
    return img

def clamp(v):
    for i in range(0,3):
        if v[i] > 255:
            v[i] = 255
        elif v[i] < 0:
            v[i] = 0
    return v

def main():
    # Format [L,M.S]
    neutral = np.asarray([1.027,0.9847,0.9182]) # Equal energy white
    a1 = np.asarray([0.05235866, 0.14667038, 0.95667258]) # 475nm adapted to D65
    a2 = np.asarray([0.9847601,0.87614013,0.00165276]) # 575nm adapted to D65
    p1, p2 = derive_planes(neutral, a1, a2)
    input_file_path, output_file_path = 'flower.jpg', 'simulated.jpg'
    img = read_image(input_file_path)
    img = simulate_protanopia(p1, p2, neutral, img)
    write_image(output_file_path, img)


if __name__ == '__main__':
    main()
