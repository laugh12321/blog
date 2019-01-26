---
title: '阿里云服务器ECS Ubuntu16.04 + Seafile 搭建私人网盘 （Seafile Pro）'
layout: post
tags:
  - Ubuntu
  - ECS
  - 服务器
  - Seafile
category: 
  - Ubuntu
  - ECS
  - Seafile
---

本文主要讲述 使用 Ubuntu 16.04 云服务器 通过脚本实现对 Seafile Pro 的安装，完成私人网盘的搭建

首先给出 Seafile 专业版的下载地址（Linux）: 👉 [传送门](https://download.seafile.com/d/6e5297246c/?p=/pro&mode=list)

在本机下载好安装包后，通过 [WinSCP](https://winscp.net/eng/download.php) 将安装包放在 `/opt/ ` 目录下，并将专业版的安装包重命名为 `seafile-pro-server_6.3.11_x86-64.tar.gz` 的格式（方便安装）。这里使用的安装方式是使用官方给出的 [Seafile 安装脚本](https://github.com/haiwen/seafile-server-installer-cn) 安装，优点是一步到位，坏处是安装失败需要还原到镜像。

<!--more-->

### 使用步骤

安装干净的 16.04 或 CentOS 7 系统，并**做好镜像** (如果安装失败需要还原到镜像)。

切换成 root 账号 (sudo -i)

#### 获取安装脚本

Ubuntu 16.04（适用于 6.0.0 及以上版本）:

```
wget https://raw.githubusercontent.com/haiwen/seafile-server-installer-cn/master/seafile-server-ubuntu-16-04-amd64-http
```

#### 运行安装脚本并指定要安装的版本 (6.3.11)

```
bash seafile-server-ubuntu-16-04-amd64-http 6.3.11
```

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Seafile/Seafile_00.png)

输入 `2` 选择安装专业版

该脚本运行完后会在命令行中打印配置信息和管理员账号密码，请仔细阅读。(你也可以查看安装日志 `/opt/seafile/aio_seafile-server.log` )，MySQL 密码在 `/root/.my.cnf` 中。

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Seafile/Seafile_01.png)

#### 通过 Web UI 对服务器进行配置

安装完成后，需要通过 Web UI 服务器进行基本的配置，以便能正常的从网页端进行文件的上传和下载：

1. 首先在浏览器中输入服务器的地址，并用管理员账号和初始密码登录

2. 点击界面的右上角的头像按钮进入管理员界面

    ![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Seafile/Seafile_04.png)

3. 进入设置页面填写正确的服务器对外的 SERVICE_URL 和 FILE_SERVER_ROOT，比如

    ```
    SERVICE_URL: http://126.488.125.111：8000
    FILE_SERVER_ROOT: 'http://126.488.125.111/seafhttp'
    ```

   <strong><font color='red'>注意：</font></strong> `126.488.125.111` 是你服务器的公网 `ip`

对了，还要在还要在 `云服务器管理控制台` 设置新的安全组规则（`8082` 和 `80` 端口），可以参考下图自行配置

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Seafile/Seafile_09.png)

现在可以退出管理员界面，并进行基本的测试。关于服务器的配置选项介绍和日常运维可以参考 <http://manual-cn.seafile.com/config/index.html>

#### 在本地打开 Web UI

因为使用一键安装脚本安装，默认使用了 `nginx` 做反向代理，并且开启了防火墙，所以你需要直接通过 `80` 端口访问，而不是 `8000` 端口。

<strong><font color='red'>注意：</font></strong>在本地输入的 `ip` 地址是你的云服务器的公网 `IP`

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Seafile/Seafile_05.png)



#### 通过客户端登陆

##### Windows 客户端登陆

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Seafile/Seafile_06.png)

##### Android 客户端登陆

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Seafile/Seafile_08.jpg)

---

至此，私人云盘已经搭建完毕

更多详细步骤请阅读： 👉 [官方脚本说明](https://github.com/haiwen/seafile-server-installer-cn)