import cv2
import numpy as np


def findhomography_and_concat(slika1, slika2):
    sift = cv2.SIFT_create(nfeatures=2000)  # nfeatures=2000 koliko ficera da trazi

    kp1, ds1 = sift.detectAndCompute(slika1, None)  # kljucne tacke i deskriptori sa 2000 ficera
    kp2, ds2 = sift.detectAndCompute(slika2, None)  # kljucne tacke i deskriptori sa 2000 ficera

    # nalazi slicnost izmedju dve slike (poklapanje ficera)
    FLANN_INDEX_KDTREE = 1  # za poklapanje se koristi KD tree zato sto dobro radi sa 2d i 3d slikama/objektima
    index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                        trees=5)  # dict kljuceva koji ce da idu FlannBasedMatcher koristim 5 kd stabala
    search_params = dict(checks=50)  # 50 iteracija
    flann = cv2.FlannBasedMatcher(index_params, search_params)  # skladistim sva poklapanja
    matches = flann.knnMatch(ds1, ds2,
                             k=2)  # poklapam flann objekat da poklopim ficere deskriptora dve slike, ovo k=2 vraca najbolje poklopljeni par

    good = []  # niz za dobre pogotke
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # razdaljina izmedju dva deskriptora m i n, ako je udaljenost mxn pomnozena sa 0.7 veca od prethodnog poklapanja, prvo se uzima
            good.append(m)

    MIN_MATCH_COUNT = 22  # proverava dobra poklapanja i vraca kljucne tacke pomocu numpy biblioteke
    # zatim racuna homografsku matricu na osnovu vrednosti iz ta dva niza
    if len(good) > MIN_MATCH_COUNT:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        result = concatImages(slika2, slika1, M)
        return result
    else:
        return False


def concatImages(slika1, slika2, H):
    rows1, cols1 = slika1.shape[:2]
    rows2, cols2 = slika2.shape[:2]

    # velicine slike1 *4 coska*
    points_list_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    # velicine slike2 *4 coska*
    points_tmp = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    # transformaciona matrica
    points_list_2 = cv2.perspectiveTransform(points_tmp, H)
    # points_list sluzi da izracuna velicinu izlazne slike, to radi tako sto uzima min i max kooridante iz points_list
    points_list = np.concatenate((points_list_1, points_list_2), axis=0)
    # distance koje treba da se primene prilikom translacije (gde se postavlja koja slika od ulaznih)
    [x_min, y_min] = np.int32(points_list.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points_list.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    # nova translaciona distanca za onu 2 i trecu sliku se kreira
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    # warpPerspective se koristi da spoji sve tako sto koristi novu translacionu matricu i poslednju sliku kako bi odredio pozicije
    output_image = cv2.warpPerspective(slika2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_image[translation_dist[1]:rows1 + translation_dist[1],
    translation_dist[0]:cols1 + translation_dist[0]] = slika1
    # gornja poslednja linija pred return sluzi da podesi velicinu izlazne slike, ovde je poravnavamo sa slika1 vrednoscu

    return output_image


# POZIV
slika1 = cv2.imread("3.JPG")
slika2 = cv2.imread("1.JPG")
slika3 = cv2.imread("2.JPG")
output_image = None

slika4 = findhomography_and_concat(slika1, slika2)
if slika4 is not False:
    output_image = findhomography_and_concat(slika4, slika3)
slika4 = findhomography_and_concat(slika1, slika3)
if slika4 is not False:
    output_image = findhomography_and_concat(slika4, slika2)
cv2.imwrite("output.jpg", output_image)
