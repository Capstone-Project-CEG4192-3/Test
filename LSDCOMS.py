
        #add line to our lines list
        #check intersection between saved lines (or we can use a function to detect if two lines are similar and save just one of the two)
        comp=comp+[line_intersection(lst1[-1],lst2[-1]) for lst1 in bigLines for lst2 in bigLines if line_intersection(lst1[-1],lst2[-1])!=False]
        #draw img
        img2 = cv2.imread(file)   
        #draw filtred line in opencv
        cv2.line(img2, [int(nx[0]),int(ny[0])], [int(nx[-1]),int(ny[-1])], (0, 0, 255), 2)
        #plt.scatter(dx,dy)
        cv2.imshow("LSD",img2 )
          
        print(f'> Line  {ff+1}/{len(lines)} : y0: {equa[-2]} slope: {det}  equa : y={det:.2}x+{det+y0:.2}')
        print(f'>Intersection points before set  : ',(len(comp)))
        if len(comp)>=1:
            comp=[ (int(xx),int(y)) for xx,y in comp]
            comp=set(comp)
            comp=list(comp)
            print(f'>Intersection points after set  : ',(len(comp)))
        #sort list of intersection according to : 
        #distance from origin
        comp.sort(key=lambda x:math.sqrt(x[0]**2+x[1]**2))
        #ratio distance/angle
        #comp.sort(key=lambda x:math.sqrt(x[0]**2+x[1]**2)/(math.atan(x[1]/x[0])))
        #actual point angle with origin
        ang1=0
        #previous point angle with origin
        ang2=0
        #actual distance btwn point and origin 
        ccs1=0
        #previous distance btwn point and origin 
        ccs2=0
        #previous point
        prevP=[0,0]
        #savedpoints
        accumPointList=[]
    #    plt.pause(0.1)
        #input()
        #if there is intersection points
    #    if len(comp)>=1:
    #      plt.scatter([c[0] for c in comp],[c[1] for c in comp],color='r')
            
        #vvv is index and ccs is actual point in the list of intersection points
        '''
        '''
            for vvv ,ccs in enumerate(comp):
                #we get distance and angle
                ccs1=math.sqrt(ccs[0]**2+ccs[1]**2)
                ang1=math.atan(ccs[1]/ccs[0])*180/math.pi
                #print actual point infos
                print(f'> Intersect Point  {vvv+1}/{len(comp)} :  ',ccs,f'   -- distance: {ccs1}  -- angle : {ang1}')
                #IF Statement to tell if actual point is the same as previois point as the list of points is sorted (testing it to see if it works)
                #we check distance btwn points + angle difference and x and y differences 
                #we can test cross product if 2 points are on same postion matbe cross product close to 0
                print(f'>cross : {np.cross(ccs,prevP)}')
                if abs(ccs1-ccs2)<20 and  abs(ang1-ang2)<10 and abs(ccs[0]-prevP[0])<10 and abs(ccs[1]-prevP[1])<10 :
                    #if two close points add then to the list accumulator 
                    accumPointList.append(ccs)
                    #if the actual point and previous point are not close we draw the actual saved points as one point since they are all similar 
                else:
                    #check if we have a lot of points that spot the same point position (generaly >2 or >5)
                    if len(accumPointList)>2:
                        #SOME Np results to compare
                        average=np.average(accumPointList,axis=0)
                        mediane=np.median(accumPointList,axis=0)
                        mean=np.mean(accumPointList,axis=0)
                        #using median for now
                        graal=mediane
                        print('>$$$$$$$ Average   :', average)
                        print('>$$$$$$$ Mediane   :', mediane)
                        print('>$$$$$$$ Mean   :', mean)
                        print('>$$$$$$$ Used   :', graal)
                        print('>$$$$$$$ SPACEMatrix at this position :' ,{space[int(graal[1]),int(graal[0])]})
                        #maybe we can use this alsp : 
                        print(f'> %%%%   %%%% Result  {len(accumPointList)}/{len(comp)} Points pointing to => {graal}')
                        #we draw the point in matplotlib
                        plt.scatter(int(graal[0]),int(graal[1]))
                        #reset acumulator list for a new point
                        accumPointList=[]
                        #split line for next Point to draw
                        print('-----------------------------------')
                        plt.pause(1)
                #we save actual point info
                prevP=ccs
                ccs2=math.sqrt(ccs[0]**2+ccs[1]**2)
                ang2=math.atan(ccs[1]/ccs[0])*180/math.pi
       
