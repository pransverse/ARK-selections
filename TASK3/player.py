from utils import Player, WINDOW_WIDTH
import cv2
import numpy as np


def template_matching(image, template):
    # Load the image and template
    img = image
    temp = template

    # Perform template matching
    result = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)

    # the get the best match fast use this:
    (min_x, max_y, minloc, maxloc) = cv2.minMaxLoc(result)
    (x, y) = minloc

    result2 = np.reshape(result, result.shape[0] * result.shape[1])
    sort = np.argsort(result2)

    # returning 3 best matches
    return np.array(
        [
            np.unravel_index(sort[0], result.shape),
            np.unravel_index(sort[1], result.shape),
            np.unravel_index(sort[2], result.shape),
        ]
    )


player = Player()
# Initializing a Player object with a random start position on a randomly generated Maze


def strategy():
    # This function is to localize the position of the newly created player with respect to the map
    Map = player.getMap()

    # template matching the original position
    Snap = player.getSnapShot()
    cv2.imwrite("OrigSnap.jpg", 255 * Snap)
    cv2.waitKey(2000)
    orig = template_matching(Map, Snap)

    # template matching after moving horizontally
    to_right = player.move_horizontal(20)
    Snap = player.getSnapShot()
    right = template_matching(Map, Snap)
    right = right - to_right

    to_left = player.move_horizontal(-20)
    Snap = player.getSnapShot()
    left = template_matching(Map, Snap)
    left = left - to_left

    # template matching after moving vertically
    to_up = player.move_vertical(20)
    Snap = player.getSnapShot()
    up = template_matching(Map, Snap)
    up = up - to_up

    to_down = player.move_vertical(-20)
    Snap = player.getSnapShot()
    down = template_matching(Map, Snap)
    down = down - to_down

    results = [0, 0, 0]
    for i in range(3):
        if orig[i].all() == right[0].all():
            results[i] += 1
        if orig[i].all() == left[0].all():
            results[i] += 1
        if orig[i].all() == up[0].all():
            results[i] += 1
        if orig[i].all() == down[0].all():
            results[i] += 1

    # Draw a rectangle around the result
    top_left = (
        orig[results.index(max(results))][1],
        orig[results.index(max(results))][0],
    )
    # top_left = orig
    h, w = Snap.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(Map, top_left, bottom_right, (0, 0, 255), 2)

    # Display the result
    cv2.imshow("Template Matching Result", Map)
    cv2.imwrite("result.jpg", 255 * Map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    strategy()
