import os
import numpy as np
# from preprocessImages import preprocessImages


def processData(inp_path):
    X=[]
    Y=[]

    i = 1
    for dirname in sorted( os.listdir(inp_path)):
        for _, _, images in os.walk(inp_path+'/'+dirname):
            for image_path in sorted(images):
                inputImage = inp_path + '/'+ dirname + '/' +image_path
                # mask = inp_path+folder+'/Msk/'+dirname + '/'+image_path
                img = preprocessImages(inputImage )
                label = i
                #storing label-1 for creating a (1,9) shaped label
                X.append(np.array(img))
                Y.append(label-1)
            i = i+1
    
    X = np.array(X).reshape(-1, 28, 28, 1)
    Y=np.array(Y).reshape(-1, 1)
    X = X/255.0   
    return X,Y
