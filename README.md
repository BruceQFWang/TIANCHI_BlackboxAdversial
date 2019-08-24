# TIANCHI_BlackboxAdversial




## 2019/8/22
测试I-FGSM，训练irse和arcface的

## 2019/8/23
集成irse的模型

**TODO**
- [x] 全脸mask
- [ ] attention mask
- [ ] 模型级联
- [x] 收集数据，对抗训练
- [ ] 多尝试猜测模型

## 2019/8/24
- [x] 由返回的L2距离减去本地计算的样本平均距离 * 712 / 44.1673  从而得到未攻击成功图片数量  (有偏估计，计算会偏多)
