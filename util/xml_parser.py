#################################
# Code taken as is from _______ #
#################################

import xml.etree.ElementTree as ET
import numpy as np

def parse_frame_dump(xml_file):
    '''parse xml file to get welding spots and torch poses'''
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    total_info = [] # list of all infos about the torch, welding spots and the transformation matrix
    

    for SNaht in root.findall('SNaht'):
        
        torch = [SNaht.get('Name'), SNaht.get('ZRotLock'), SNaht.get('WkzWkl'), SNaht.get('WkzName')]
        weld_frames = [] # list of all weld_frames as np.arrays(X,Y,Z) in mm
        pose_frames = [] # list of all pose_frames as 4x4 homogenous transforms
        
        for Kontur in SNaht.findall('Kontur'):  
            for Punkt in Kontur.findall('Punkt'):
                X = float(Punkt.get('X'))
                Y = float(Punkt.get('Y'))
                Z = float(Punkt.get('Z'))
                Norm = []
                for Fl_Norm in Punkt.findall('Fl_Norm'):
                    Norm_X = float(Fl_Norm.get('X'))
                    Norm_Y = float(Fl_Norm.get('Y'))
                    Norm_Z = float(Fl_Norm.get('Z'))
                    Norm.append(np.array([Norm_X, Norm_Y, Norm_Z]))
                weld_frames.append({'position': np.array([X, Y, Z]), 'norm': Norm})

        # desired model output
        for Frames in SNaht.findall('Frames'):  
            for Frame in Frames.findall('Frame'):
                torch_frame = np.zeros((4,4))  # 4x4 homogenous transform
                torch_frame[3,3] = 1.0

                for Pos in Frame.findall('Pos'):
                    # 3x1 position
                    X = Pos.get('X')
                    Y = Pos.get('Y')
                    Z = Pos.get('Z')
                    torch_frame[0:3,3] = np.array([X,Y,Z])
                for XVek in Frame.findall('XVek'):
                    # 3x3 rotation
                    Xrot = np.array([XVek.get('X'), XVek.get('Y'), XVek.get('Z')])      
                    torch_frame[0:3, 0] = Xrot
                for YVek in Frame.findall('YVek'):
                    # 3x3 rotation
                    Yrot = np.array([YVek.get('X'), YVek.get('Y'), YVek.get('Z')])      
                    torch_frame[0:3, 1] = Yrot
                for ZVek in Frame.findall('ZVek'):
                    # 3x3 rotation
                    Zrot = np.array([ZVek.get('X'), ZVek.get('Y'), ZVek.get('Z')])      
                    torch_frame[0:3, 2] = Zrot

                #print(torch_frame) 
                pose_frames.append(torch_frame)
        
        total_info.append({'torch': torch, 'weld_frames': weld_frames, 'pose_frames': pose_frames})
        
    return total_info


def parse_xml_to_array(xml_file):
    '''parse xml file to get welding spots and torch poses'''
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    total_info = [] # list of all infos about the torch, welding spots and the transformation matrix
    

    for SNaht in root.findall('SNaht'):
        weld_info = []
        weld_info.append(SNaht.get('Name'))
        weld_info.append(SNaht.get('ZRotLock'))
        weld_info.append(SNaht.get('WkzWkl'))
        torch = SNaht.get('WkzName')
        if torch == 'MRW510_10GH':
            weld_info.append(0)
        elif torch  == 'TAND_GERAD_DD':
            weld_info.append(1)
        else:
            weld_info.append(2)
        
        for Kontur in SNaht.findall('Kontur'):  
            for Punkt in Kontur.findall('Punkt'):
                X = float(Punkt.get('X'))
                Y = float(Punkt.get('Y'))
                Z = float(Punkt.get('Z'))
                weld_info.append(X)
                weld_info.append(Y)
                weld_info.append(Z)

                for Fl_Norm in Punkt.findall('Fl_Norm'):
                    Norm_X = float(Fl_Norm.get('X'))
                    Norm_Y = float(Fl_Norm.get('Y'))
                    Norm_Z = float(Fl_Norm.get('Z'))
                    weld_info.append(Norm_X)
                    weld_info.append(Norm_Y)
                    weld_info.append(Norm_Z)

        for Frames in SNaht.findall('Frames'):  
            for Frame in Frames.findall('Frame'):
                for XVek in Frame.findall('XVek'):
                    # 3x3 rotation
                    weld_info.append(XVek.get('X'))
                    weld_info.append(XVek.get('Y'))
                    weld_info.append(XVek.get('Z'))
                for YVek in Frame.findall('YVek'):
                    # 3x3 rotation
                    weld_info.append(YVek.get('X'))
                    weld_info.append(YVek.get('Y'))
                    weld_info.append(YVek.get('Z'))  
                for ZVek in Frame.findall('ZVek'):
                    # 3x3 rotation
                    weld_info.append(ZVek.get('X'))
                    weld_info.append(ZVek.get('Y'))
                    weld_info.append(ZVek.get('Z'))  
        
        total_info.append(np.asarray(weld_info))
    return np.asarray(total_info)

def list2array(total_info):
    res = []
    for info in total_info:
        for i, spot in enumerate(info['weld_frames']):
            weld_info = []
            weld_info.append(info['torch'][0])
            weld_info.append(info['torch'][1])
            weld_info.append(info['torch'][2])
            torch = info['torch'][3]
            if torch == 'MRW510_10GH':
                weld_info.append(0)
            elif torch  == 'TAND_GERAD_DD':
                weld_info.append(1)
            else:
                weld_info.append(2)
            weld_info.append(spot['position'][0])
            weld_info.append(spot['position'][1])
            weld_info.append(spot['position'][2])
            weld_info.append(spot['norm'][0][0])
            weld_info.append(spot['norm'][0][1])
            weld_info.append(spot['norm'][0][2])
            weld_info.append(spot['norm'][1][0])
            weld_info.append(spot['norm'][1][1])
            weld_info.append(spot['norm'][1][2])
            weld_info.append(info['pose_frames'][i][0][0])
            weld_info.append(info['pose_frames'][i][1][0])
            weld_info.append(info['pose_frames'][i][2][0])
            weld_info.append(info['pose_frames'][i][0][1])
            weld_info.append(info['pose_frames'][i][1][1])
            weld_info.append(info['pose_frames'][i][2][1])
            weld_info.append(info['pose_frames'][i][0][2])
            weld_info.append(info['pose_frames'][i][1][2])
            weld_info.append(info['pose_frames'][i][2][2])
            res.append(np.asarray(weld_info))
    return np.asarray(res)



if __name__== '__main__':
    t = parse_frame_dump('./data_sep/predictions/201910204483_R4.xml')
    # g = parse_frame_dump('./data_sep/models/201910204483/201910204483_R1.xml')
    r1 = list2array(t)
    r2 = parse_xml_to_array('./data_sep/models/201910204483/201910204483_r.xml')
    print (r1[0])
    print ('............................')
    print (r2[0])

