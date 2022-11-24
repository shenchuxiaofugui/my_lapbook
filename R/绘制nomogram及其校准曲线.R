library(foreign) # 用于读取外部数据的包，和readr一个用途
library(survival)
library(rms)
library(ResourceSelection)
# 需要准备用于训练的数据csv文件
# 1.改文件路径
# 2.改掉用于训练的每个特征名称
# 3.如果要画测试结果，就改test路径
# 用于训练的数据路径
traindata<-read.csv('C:/Users/HJ Wang/Desktop/nomo/RadiomicClinicoradiographic.csv')
# 如果需要画测试结果的校准曲线
testdata<-read.csv('C:/Users/HJ Wang/Desktop/nomo/RadiomicClinicoradiographic.csv')
# 用于训练的label和feature名称

formula1<-as.formula(Label~CT_original_firstorder_90Percentile
                     +CT_original_glcm_ClusterShade
                     +CT_original_glcm_JointAverage
                     +CT_original_glszm_SmallAreaHighGrayLevelEmphasis
                     +CT_original_shape_SurfaceVolumeRatio)


# ---------------------------------------------------------------
# 对数据打包，载入环境中,就不用一直重复访问数据了
dd <- datadist(traindata)
options(datadist='dd')
# 建立回归模型
model <- lrm(formula1, data = traindata, x=T,y=T, tol=1e-9)
# 解析模型
summary(model)
# 绘制列线图
nomomodel<-nomogram(model,
                    fun=function(x)1/(1+exp(-x)),
                    lp=F,
                    # 分几个数据点显示
                    fun.at = c(0.001,0.01,0.05,0.1,0.3,0.5,0.7,0.9,0.95,0.98),
                    # 设置最后输出行的名称
                    funlabel = "Probability")

plot(nomomodel, cex.var = 2.2, cex.axis = 2.2)

# 再画校准曲线
calibrationmodel<-calibrate(model,method = 'boot', B=1000)
calibrationmodel
plot(calibrationmodel,xlim=c(0,1.0),ylim=c(0,1.0),
     # XY轴的名称
     xlab = "Nomogram Predicted Probability",
     ylab = "Actual Probability",
     # cex表示字体大小
     cex=1, cex.lab=1.2, cex.axis=1, cex.main=1.2)

# 校准曲线画完之后还有计算个HOSMER-LEMESHOW拟合优度检验
# 表示校准曲线拟合的怎么样是不是可以接受的，得到一个P值，P越大表示不能拒绝这个拟合
train_hl <- hoslem.test(traindata$label, predict(object = model,type = "fitted"), g=10)
train_hl
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 根据建模组方程，计算test预测概率
dd <- datadist(testdata)
options(datadist='dd')
valpro<-predict(object = model,type = "fitted",newdata = testdata)
valpro
# 根据结果建立回归方程
dd <- datadist(valpro)
options(datadist='dd')
testmodel<-lrm(label~valpro, data=testdata, x=T, y=T)
summary(testmodel)
cal2 <- calibrate(testmodel, method='boot', B=1000,data=testdata)
par(cex=1.7, cex.lab=1.2, cex.axis=1.2, lwd=2, no.readonly = FALSE)
plot(cal2,xlim=c(0,1.0),ylim=c(0,1.0),
     # XY轴的名称
     xlab = "Validation of T2 Nomogram Predicted Probability",
     ylab = "Actual Probability")

# 校准曲线画完之后还有计算个HOSMER-LEMESHOW拟合优度检验
# 表示校准曲线拟合的怎么样是不是可以接受的，得到一个P值，P越大表示不能拒绝这个拟合
test_hl <- hoslem.test(testdata$label, valpro, g=10)
test_hl
cbind(test_hl$observed,test_hl$expected)
# ------------------------------------------------------------------
# 这是另一种test结果的验证图
# 先预测score再转换logistic回归
# 建立回归模型
pred.logit = predict(model, newdata = testdata)
y_pred <- 1/(1+exp(-pred.logit))
val.prob(y_pred,
         as.numeric(testdata$label),
         m=20,cex=1)

