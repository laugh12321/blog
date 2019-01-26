---
title: 'Jekyll + NexT + GitHub Pages 主题深度优化'
layout: post
tags:
  - Jekyll
  - Blog
  - GitHub Pages
  - Next
category: 
  - Jekyll
  - Blog
  - GitHub Pages
  - Next
---

# 前言
笔者在用 Jekyll 搭建个人博客时踩了很多的坑，最后发现了一款不错的主题 [jekyll-theme-next](https://github.com/simpleyyt/jekyll-theme-next)，但网上关于 Jekyll 版的 Next 主题优化教程少之又少，于是就决定自己写一篇以供参考。

> 本文仅讲述 Next (Jekyll) 主题的深度优化操作，关于主题的基础配置请移步[官方文档](http://theme-next.simpleyyt.com/)。

<!--more-->

# 主题优化

## 修改内容区域的宽度

打开 `_sass/_custom/custom.scss` 文件，新增变量：

```scss
// 修改成你期望的宽度
$content-desktop = 700px

// 当视窗超过 1600px 后的宽度
$content-desktop-large = 900px
```

> 此方法不适用于 Pisces Scheme

当你使用Pisces风格时可以用下面的方法：

- 编辑 Pisces Scheme 的  `_sass/_schemes/Pisces/_layout.scss` 文件，在最底部添加如下代码：

  ```scss
  header{ width: 90%; }
  .container .main-inner { width: 90%; }
  .content-wrap { width: calc(100% - 260px); }
  ```

  > 对于有些浏览器或是移动设备，效果可能不是太好

- 编辑 Pisces Scheme 的  `_sass/_schemes/Pisces/_layout.scss` 文件，修改以下内容：

  ```scss
  // 将 .header 中的 
  width: $main-desktop;
  // 改为：
  width: 80%;
  
  // 将 .container .main-inner 中的：
  width: $main-desktop;
  // 改为：
  width: 80%;
  
  // 将 .content-wrap 中的：
  width: $content-desktop;
  // 改为：
  width: calc(100% - 260px);
  ```

  > 还是不知道如何修改的话，可以直接赋值笔者改好的 👉 [传送门](https://github.com/laugh12321/blog/blob/master/_sass/_schemes/Pisces/_layout.scss)


## 背景透明度

打开 `_sass/_custom/custom.scss` 文件，新增变量：

```scss
//文章内容背景改成了半透明
.content-wrap {
    background: rgba(255, 255, 255, 0.8);
}
.sidebar {
    background: rgba(255, 255, 255, 0.1);
    box-shadow: 0 2px 6px #dbdbdb;
}
.site-nav{
    box-shadow: 0px 0px 0px 0px rgba(0, 0, 0, 0.8);
}
.sidebar-inner {
	background: rgba(255, 255, 255, 0.8);
	box-shadow: 0 2px 6px #dbdbdb;
}
.header-inner {
    background: rgba(255, 255, 255, 0.8);
    box-shadow: 0 2px 6px #dbdbdb;
}
.footer {
    font-size: 14px;
    color: #434343;
}
```


## 自定义背景图片

打开 `_sass/_custom/custom.scss` 文件，新增变量：

```scss
body{
    background:url(https://images8.alphacoders.com/929/929202.jpg);
    background-size:cover;
    background-repeat:no-repeat;
    background-attachment:fixed;
    background-position:center;
}
```

> `url()` 中可以时本地图片，也可以是图片链接



## 彩色时间轴

打开 `_sass/_custom/custom.scss` 文件，新增变量：

```scss
// 时间轴样式
.posts-collapse {
    margin: 50px 0px;
}
@media (max-width: 1023px) {
    .posts-collapse {
        margin: 50px 20px;
    }
}

// 时间轴左边线条
.posts-collapse::after {
    margin-left: -2px;
    background-image: linear-gradient(180deg,#f79533 0,#f37055 15%,#ef4e7b 30%,#a166ab 44%,#5073b8 58%,#1098ad 72%,#07b39b 86%,#6dba82 100%);
}

// 时间轴左边线条圆点颜色
.posts-collapse .collection-title::before {
    background-color: rgb(255, 255, 255);
}

// 时间轴文章标题左边圆点颜色
.posts-collapse .post-header:hover::before {
    background-color: rgb(161, 102, 171);
}

// 时间轴年份
.posts-collapse .collection-title h1, .posts-collapse .collection-title h2 {
    color: rgb(102, 102, 102);
}

// 时间轴文章标题
.posts-collapse .post-title a {
    color: rgb(80, 115, 184);
}
.posts-collapse .post-title a:hover {
    color: rgb(161, 102, 171);
}

// 时间轴文章标题底部虚线
.posts-collapse .post-header:hover {
    border-bottom-color: rgb(161, 102, 171);
}

// archives页面顶部文字
.page-archive .archive-page-counter {
    color: rgb(0, 0, 0);
}

// archives页面时间轴左边线条第一个圆点颜色
.page-archive .posts-collapse .archive-move-on {
    top: 10px;
    opacity: 1;
    background-color: rgb(255, 255, 255);
    box-shadow: 0px 0px 10px 0px rgba(0, 0, 0, 0.5);
}
```

## 友链居中

打开 `_sass/_custom/custom.scss` 文件，新增变量：

```scss
//友链居中
.links-of-blogroll-title {
  text-align: center;
}
```

## 修改友链文本颜色

打开 `_sass/_custom/custom.scss` 文件，新增变量：

```scss
//友链文本颜色
//将链接文本设置为蓝色，鼠标划过时文字颜色加深，并显示下划线
.post-body p a{
  text-align: center;
  color: #434343;
  border-bottom: none;
  &:hover {
    color: #5073b8;
    text-decoration: underline;
  }
}
```

## 修改友链样式

打开 `_sass/_custom/custom.scss` 文件，新增变量：

```scss
//修改友情链接样式
.links-of-blogroll-item a{
  text-align: center;
  color: #434343;
  border-bottom: none;
  &:hover {
    color: #5073b8;
    text-decoration: underline;
  }
}
```

## 自定义页脚的心样式

打开 `_sass/_custom/custom.scss` 文件，新增变量：

```scss
// 自定义页脚的心样式
@keyframes heartAnimate {
    0%,100%{transform:scale(1);}
    10%,30%{transform:scale(0.9);}
    20%,40%,60%,80%{transform:scale(1.1);}
    50%,70%{transform:scale(1.1);}
}
#heart {
    animation: heartAnimate 1.33s ease-in-out infinite;
}
.with-love {
    color: rgb(255, 113, 168);
}
```

## 设置头像边框为圆形框

打开 `_sass/_common/components/sidebar/sidebar-author.scss` 文件，新增变量：

```scss
.site-author-image {
  display: block;
  margin: 0 auto;
  padding: $site-author-image-padding;
  max-width: $site-author-image-width;
  height: $site-author-image-height;
  border: $site-author-image-border-width solid $site-author-image-border-color;
  
  /* 头像圆形 */
  border-radius: 80px;
  -webkit-border-radius: 80px;
  -moz-border-radius: 80px;
  box-shadow: inset 0 -1px 0 #333sf;
```

## 特效：鼠标放置头像上旋转

打开 `_sass/_common/components/sidebar/sidebar-author.scss` 文件，新增变量：

```scss
  /* 设置循环动画 [animation: (play)动画名称 (2s)动画播放时长单位秒或微秒 (ase-out)动画播放的速度曲线为以低速结束 
    (1s)等待1秒然后开始动画 (1)动画播放次数(infinite为循环播放) ]*/
 

  /* 鼠标经过头像旋转360度 */
  -webkit-transition: -webkit-transform 1.0s ease-out;
  -moz-transition: -moz-transform 1.0s ease-out;
  transition: transform 1.0s ease-out;
}

img:hover {
  /* 鼠标经过停止头像旋转 
  -webkit-animation-play-state:paused;
  animation-play-state:paused;*/

  /* 鼠标经过头像旋转360度 */
  -webkit-transform: rotateZ(360deg);
  -moz-transform: rotateZ(360deg);
  transform: rotateZ(360deg);
}

/* Z 轴旋转动画 */
@-webkit-keyframes play {
  0% {
    -webkit-transform: rotateZ(0deg);
  }
  100% {
    -webkit-transform: rotateZ(-360deg);
  }
}
@-moz-keyframes play {
  0% {
    -moz-transform: rotateZ(0deg);
  }
  100% {
    -moz-transform: rotateZ(-360deg);
  }
}
@keyframes play {
  0% {
    transform: rotateZ(0deg);
  }
  100% {
    transform: rotateZ(-360deg);
  }
}
```

# Bug 修复

## 打赏文字抖动修复

打开  `_sass/_common/components/post/post-reward.scss` 文件，然后注释其中的函数 `wechat:hover` 和 `alipay:hover` ，如下：

```scss
/*
  #wechat:hover p{
      animation: roll 0.1s infinite linear;
      -webkit-animation: roll 0.1s infinite linear;
      -moz-animation: roll 0.1s infinite linear;
  }
  #alipay:hover p{
      animation: roll 0.1s infinite linear;
      -webkit-animation: roll 0.1s infinite linear;
      -moz-animation: roll 0.1s infinite linear;
  }
*/
```

## 修改文章底部的带#号标签

打开 `_includes/_macro/post.html` 文件,搜索 `rel="tag">#` ,将 `#` 换成 `<i class="fa fa-tag"></i>`

```html
<div class="post-tags">
    {% for tag in post.tags %}
    {% assign tag_url_encode = tag | url_encode | replace: '+', '%20' %}
    <a href="{{ '/tag/#/' | relative_url | append: tag_url_encode }}" rel="tag"><i class="fa fa-tag"></i> {{ tag }}</a>
    {% endfor %}
</div>
```

# 插件配置

## 阅读次数统计（LeanCloud）

- 请查看 [为NexT主题添加文章阅读量统计功能](https://notes.wanghao.work/2015-10-21-%E4%B8%BANexT%E4%B8%BB%E9%A2%98%E6%B7%BB%E5%8A%A0%E6%96%87%E7%AB%A0%E9%98%85%E8%AF%BB%E9%87%8F%E7%BB%9F%E8%AE%A1%E5%8A%9F%E8%83%BD.html#%E9%85%8D%E7%BD%AELeanCloud)

- 打开 `config.yml` 文件，搜索 `leancloud_visitors` , 进行如下更改：

  ```yml
  leancloud_visitors:
    enable: true
    app_id: <app_id>
    app_key: <app_key>
  ```

  > `app_id` 和 `app_key` 分别是 你的LearnCloud 账号的 `AppID` 和 `AppKey`

## 阅读次数美化

效果👉：![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Jekyll-theme-next_00.png)

- 打开 `_data/languages/zh-Hans.yml` 文件，将 `post` 中的 `visitors:阅读次数 ` 改为：`visitors: 热度`。
- 打开 `_includes/_macro/post.html` 文件,搜索 `leancloud-visitors-count` ,在 `<span></span>` 之间添加 `℃`

  ```html
  <span id="{{ post.url | relative_url }}" class="leancloud_visitors" data-flag-title="{{ post.title }}">
    <span class="post-meta-divider">|</span>
    <span class="post-meta-item-icon">
      <i class="fa fa-eye"></i>
    </span>
    {% if site.post_meta.item_text %}
      <span class="post-meta-item-text">{{__.post.visitors}} </span>
    {% endif %}
      <span class="leancloud-visitors-count"></span>
      <span>℃</span>
  </span>
  ```

## 在网站底部加上访问量

效果👉：![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Jekyll-theme-next_01.png)

- 打开 `_includes/_partials/footer.html` 文件，在 `<div class="copyright" >` 之前加入下面的代码：

  ```html
  <script async src="//busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js"></script>
  ```

- 在 `if site.copyright ` 之后加入下面的代码：

  ```html
  <div class="powered-by">
  <i class="fa fa-user-md"></i><span id="busuanzi_container_site_uv">
    本站访客数:<span id="busuanzi_value_site_uv"></span>
  </span>
  </div>
  ```



---


>部分样式可参考我的博客👉：[Laugh's Blog](https://www.laugh12321.cn)
>
>参考文章：
>
>[Hexo+Next主题优化](http://www.dragonstyle.win/3358042383.html)
>
>[hexo的next主题个性化教程:打造炫酷网站](http://shenzekun.cn/hexo%E7%9A%84next%E4%B8%BB%E9%A2%98%E4%B8%AA%E6%80%A7%E5%8C%96%E9%85%8D%E7%BD%AE%E6%95%99%E7%A8%8B.html)
>
>参考博客：
>
>[DS Blog](https://www.ds-vip.top/)

