```
系统环境，centos
```
#### 安装R文件
1)、** 安装R之前先安装系统软件，如果已经安装则略过这一步；**
```
sudo yum install readline-devel
sudo yum install libXt-devel
sudo yum install gcc-gfortran
sudo yum install gcc-c++
sudo yum install glibc-headers
```
2)、**编译R 语言**
```
sudo ./configure --prefix=/usr/local/lib64/R --enable-R-shlib --with-readline=yes --with-libpng=yes --with-x=no
```
>prefix即为R语言的系统路径；

3)、** 安装R 语言 **
```
sudo make && make install
```
4)、建立R语言的软连接
```
> ln -s /usr/local/lib64/R/bin/R  /usr/bin/R
> ln -s /usr/locallib64/R/bin/Rscript /usr/bin/Rscript
```

### 配置Rserve
1)、** 先安装Rserve 包 **
- 官网下载Rserve包
- 安装包：
```
R CMD INSTALL Rserve_1.7-3.tar.gz
```
2)、** 启动Rserve **
- 进入R
- 执行脚本 :
```
library(Rserve);
Rserve(args="--RS-enable-remote --no-save");
```

### Issues
a) yum源问题导致R安装失败
```
/etc/yum.repos.d/下运行“grep '\[base\]' *”,找到所配置的base yum源，
```

b) 正确的yum源配置如下：
```
[base]
name=CentOS-$releasever - Base
baseurl=http://mirrors.sh.ctriptravel.com/centos/$releasever/os/$basearch/
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-6
```

其中，$releasever为"rpm -q --qf %{version} centos-release;echo"的输出，$basearch为“arch”输出
有时南通测试服务器上预安装的版本比较高，会导致mirrors.sh yum repo不能用，
需要改为http://10.8.84.41/centos/$releasever/os/$basearch/
或者 在/etc/hosts  增加 ：10.8.84.41 mirrors.sh.ctriptravel.com

Rserv配置实现负载功能
Rserv_6311.conf文件内容如下：
port 6311
source D:/Ctrip/R/calcReservedRoom.R

Rserv_6312.conf文件内容如下：
port 6312
source D:/Ctrip/R/calcReservedRoom.R

启动两个Rserver,命令如下：
在R console中输入Rserve(args="--RS-enable-remote --no-save");
Rserve(args="--RS-enable-remote --no-save  --RS-confE:/ProgramFiles/R/R-3.1.3/etc/Rserv_6311.conf")
Rserve(args="--RS-enable-remote --no-save  --RS-conf E:/ProgramFiles/R/R-3.1.3/etc/Rserv_6312.conf")
或者在shell中输入R CMD Rserve --RS-enable-remote --no-save --RS-conf E:/ProgramFiles/R/R-3.1.3/etc/Rserv_6312.conf
这样就启动了两个Rserve，分别对应两个不同的conf文件

程序中配置，RConnectionPool.xml内容如下：
<RServer host="localhost" port="6311" />
<RServer host="localhost" port="6312" />
配置两个Rserver节点，分别对应conf文件中的端口号，即可。
以上可以实现简单的负载功能，当一个Rserver无法访问时，会自动尝试第二个Rserver


以下是单机模拟方式进行的Rserve负载均衡配置：


创建两个conf文件（这里举例创建2个，可多个）配置如下：
Rserv_6311.conf文件内容如下：
port 6311
source D:/Ctrip/R/calcReservedRoom.R

Rserv_6312.conf文件内容如下：
port 6312
source D:/Ctrip/R/calcReservedRoom.R

     2. 启动两个Rserver,命令如下：
Rserve(args="--RS-conf E:/ProgramFiles/R/R-3.1.3/etc/Rserv_6311.conf")
Rserve(args="--RS-conf E:/ProgramFiles/R/R-3.1.3/etc/Rserv_6312.conf")
说明：这样就启动了两个Rserve，对应两个不同的端口号6311和6312

     3. 程序中配置，RConnectionPool.xml内容如下：
<RServer host="localhost" port="6311" />
<RServer host="localhost" port="6312" />
说明：配置两个Rserver节点，与两个conf文件中的端口号一一对应，即可。

以上可以实现简单的负载功能，当一个Rserver无法访问时，会自动尝试第二个Rserver

另：真实环境应该是多台机器有不同的ip，端口号可以相同；分别启动这几台机器的Rserve；RConnectionPool.xml中配置所有机器的ip和端口。一旦某台机器挂掉，程序会自动访问其它机器上的Rserve.


3. 测试客户端
下载datacommon.jar
运行 java -jar datacommon.jar <request> <RScript.path> <RData.path> <RConnectionPoolConfig.path>


使用java与Rserve通信：
1，下载REngine.jar和Rserve.jar #下载地址 ：http://www.rforge.net/Rserve/files/
或者将pom.xml文件导入maven依赖
<dependency>
<groupId>org.rosuda.REngine</groupId>
<artifactId>REngine</artifactId>
<version>2.1.0</version>
</dependency>
<dependency>
<groupId>org.rosuda.REngine</groupId>
<artifactId>Rserve</artifactId>
<version>1.8.1</version>
</dependency>
2，编写代码demo
