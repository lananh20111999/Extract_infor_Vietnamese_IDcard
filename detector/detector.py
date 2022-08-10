import numpy as np
def get_center_point(lista):
    di = dict()

    for index in range (len(lista)):
        xmin = lista[index][0]
        ymin = lista[index][1]
        xmax = lista[index][2]
        ymax = lista[index][3]
        name = lista[index][6]
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        di[name] = (x_center, y_center)
    # print(di)
    return di


def detectIDcard(listCorners):
	frontNew = {"newFrontTopLeft", "newFrontTopRight", "newFrontBotRight", "newFrontBotLeft"}
	backNew = {"newBackTopLeft", "newBackTopRight", "newBackBotRight", "newBackBotLeft"}
	frontOld = {"oldFrontTopLeft", "oldFrontTopRight", "oldFrontBotRight", "oldFrontBotLeft"}
	backOld = {"oldBackTopLeft", "oldBackTopRight", "oldBackBotRight", "oldBackBotLeft"}
	cropP = []
	di = get_center_point(listCorners)
	myset = set(di.keys())
	cardType = "invalid"

	if frontNew == myset:
		cropP.append(list(di['newFrontTopLeft']))
		cropP.append(list(di['newFrontTopRight']))
		cropP.append(list(di['newFrontBotRight']))
		cropP.append(list(di['newFrontBotLeft']))
		cardType =  "newFront"

	elif backNew == myset:
		cropP.append(list(di['newBackTopLeft']))
		cropP.append(list(di['newBackTopRight']))
		cropP.append(list(di['newBackBotRight']))
		cropP.append(list(di['newBackBotLeft']))
		cardType =  "newBack"

	elif frontOld == myset:
		cropP.append(list(di['oldFrontTopLeft']))
		cropP.append(list(di['oldFrontTopRight']))
		cropP.append(list(di['oldFrontBotRight']))
		cropP.append(list(di['oldFrontBotLeft']))
		cardType =  "oldFront"

	elif backOld == myset:
		cropP.append(list(di['oldBackTopLeft']))
		cropP.append(list(di['oldBackTopRight']))
		cropP.append(list(di['oldBackBotRight']))
		cropP.append(list(di['oldBackBotLeft']))
		cardType =  "oldBack"

	cropP = np.array(cropP)

	return cropP, cardType
