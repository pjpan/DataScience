add file /home/hotelbi/ppj/transformtest/splitcheckindate.py;
use tmp_htlbidb;
drop table if exists ppj_htl_test;
create table if not exists ppj_htl_test
as
select transform(orderid,orderdate,hotel,room,arrival,etd,ordroomnum,ordroomstatus,isholdroom,freesale,orderstatus)
USING  'python splitcheckindate.py'
AS  checkindate,orderid,orderdate,hotel,room,arrival,etd,ordroomnum,ordroomstatus,isholdroom,freesale,orderstatus,insertdate
FROM (
select orderid
,orderdate
,hotel
,room
,arrival
,etd
,ordroomnum,ordroomstatus,isholdroom,freesale
FROM dw_htldb.facthtlordersnap
where d>='2015-05-18'
) tt;


hive -e "set hive.cli.print.header=false;
 select orderid
,orderdate
,hotel
,room
,arrival
,etd
,ordroomnum,ordroomstatus,isholdroom,freesale,orderstatus
FROM dw_htldb.facthtlordersnap
where d='2015-11-18' limit 10;" >test.txt

select * from  tmp_htlbidb.ppj_htl_test limit 10;
