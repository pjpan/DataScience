#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:17:24 2015

@author: pjpan
"""

# inputparameter: orderid,orderdate,hotel,room,arrival,etd,ordroomnum,ordroomstatus,isholdroom,freesale
# line = '1577318385      2015-11-18      700725  15811524        2015-11-20      2015-11-22      1       U       F       F'

import sys
import time

#f = open("test.txt","r")
#lines = f.readlines()

#add inserttime
insertdate=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))

for line in sys.stdin:
        try:
            (orderid,orderdate,hotel,room,arrival,etd,ordroomnum,ordroomstatus,isholdroom,freesale,orderstatus,cancelreason,ordquantity,ciireceivable,ciiquantity,ciiroomnum)=line.strip().split('\t')
            print line
        except:
            (orderid,orderdate,hotel,room,arrival,etd,ordroomnum,ordroomstatus,isholdroom,freesale,orderstatus,cancelreason,ordquantity,ciireceivable,ciiquantity,ciiroomnum)=line.strip().split(',')
            print >> sys.stderr ,'CAN NOT split:,please check your data sequence:'+ line
            continue
            #sys.exit()
  
        checkindate = time.localtime(time.mktime(time.strptime(arrival[0:10],'%Y-%m-%d')))
        EndDate = time.localtime(time.mktime(time.strptime(etd[0:10],'%Y-%m-%d')))
        while(time.strftime("%Y-%m-%d",checkindate) < time.strftime("%Y-%m-%d",EndDate)):     
            print '\t'.join([time.strftime("%Y-%m-%d",checkindate),orderid,orderdate,hotel,room,arrival,etd,ordroomnum,ordroomstatus,isholdroom,freesale,orderstatus,insertdate,cancelreason,ordquantity,ciireceivable,ciiquantity,ciiroomnum ])
            checkindate=time.localtime(time.mktime(checkindate)+1*24*3600)
