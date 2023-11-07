

import serial
ser = serial.Serial('COM11', 9800, timeout=1)
while(True):
    line =   ser.readline().decode()
    if(line=='' or line==' '):
        print("empty")
    else:
        print("not")
        opt=int(line)
        if(opt==1):
            from I2T import m
            r=m()
            r.mainloop()
        elif(opt==2):
            from video import object_detection
            object_detection()
        elif(opt==3):
            from attendance import TrackImages
            TrackImages()
        elif(opt==4):
            from currency_detection import currency
            m=currency()
            m.mainloop()


   
    
