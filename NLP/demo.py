#-*- coding: utf-8 -*-
import sys
import thulac
import re
reload(sys)
sys.setdefaultencoding('utf-8')

#thu1 = thulac.thulac(seg_only=False, model_path="./models", rm_space=True)  #设置模式为行分词模式
#a = thu1.cut("对阻碍国家机关工作人员依法执行职务的，构成违反治安管理行为情节严重的,依照《中华人民共和国治安管理处罚法》第五十条，处五日以上十日以下拘留")
#s = 'http://photocdn.sohu.com/20160104/mp52057163_1451885296515_1_th.jpeg	山西国有土地上房屋征收与补偿条例2016年实施	　　山西拆迁律师：山西国有土地上房屋征收与补偿条例2016年实施 　　《山西省国有土地上房屋征收与补偿条例》已由山西省第十二届人民代表大会常务委员会第二十二次会议于2015年9月24日通过'
#s = 'http://img.mp.itc.cn/upload/20160629/0c52289f0712455ab47884982ccd30c3_th.jpg	美国移民值得体验的5大美国特色趣事	　　 　　1.在波士顿的芬威球场上看一场棒球赛 　　芬威球场是一片神圣的土地。这里的座位十分接近赛场，以至于观众可以听到棒球的击球声在这个有102年历史的棒球场上保持了很多传统'
#s = 'http://img.mp.itc.cn/upload/20161130/cf6f2d3b1aa64372a4febcc988f75115_th.jpeg	石林至泸西高速&amp;诏安县道路综合提升项目	　　石林至泸西高速公路（昆明段）建设项目“施工总承包+股权投资”中标公示 　　项目概况 　　一、建设地点：'
#s = 'http://n1.itc.cn/img8/wb/recom/2016/05/14/146318497905896065.jpg	夯实“一带一盟”战略对接	　　 　　本报记者近日探访北极圈内的中俄合作亚马尔液化天然气项目。图①为本报驻俄罗斯记者(右一)采访亚马尔液化天然气项目负责人(左三)。图②为该项目工人正在吊装设备。图③为中国生产的重750吨的核心模块正在组装。 　　本报记者5月5日前往位于北极圈内的中俄合作的亚马尔液化天然气'
#s = 'http://n1.itc.cn/img8/wb/recom/2016/07/01/146733558238121284.JPEG	创优争先 扶贫一线党旗红	　　 　　区纪委党员干部在李家坝村开展党群交流活动 　　在中国共产党成立95周年之际，顺庆区积极践行“两学一做”，数千名党员干部集中开展示范活动，以实际行动为党的生日献礼。 　　一顿农家饭见证党群鱼水深情'
spt = re.split('\t', s)
#print spt[2].split(' ')[1]
for string in spt:
    print string
print 'len(spt) = %d' %len(spt)
t = spt[2].decode('utf-8')
spt2 = t.split(u'\u3000')
find = False
for string in spt2:
    if len(string) and string != ' ':
        print string
        if not find:
            tag = string
        find = True
print 'len(spt2) = %d' %len(spt2)
print 'tag: %s' %tag
a = thu1.cut(s, text=False)
for each in a:
    print each[0], ' ', each[1]
