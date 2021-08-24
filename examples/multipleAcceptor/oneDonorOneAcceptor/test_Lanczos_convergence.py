import numpy as np
from fishbonett.backwardSpinBosonMultiChannel import SpinBoson
import scipy as sp
from examples.multipleAcceptor.electronicParametersAndVibronicCouplingDA import coupMol2_CT, coupMol2_LE, freqMol1_GR, freqMol2_LE
from fishbonett.lanczos import lanczos

bath_length = 162 * 2
phys_dim = 20
bond_dim = 1000

a = [phys_dim] * bath_length
pd = a[::-1] + [2]

coup_num_LE = np.array(coupMol2_LE)
coup_num_CT = np.array(coupMol2_CT)
freqMol2_LE = [18.0, 22.0, 37.0, 53.0, 60.0, 64.0, 80.0, 90.0, 113.0, 138.0, 145.0, 150.0, 164.0, 201.0, 218.0, 239.0, 281.0, 287.0, 296.0, 298.0, 324.0, 328.0, 378.0, 390.0, 398.0, 409.0, 418.0, 423.0, 454.0, 463.0, 472.0, 486.0, 494.0, 512.0, 528.0, 529.0, 536.0, 556.0, 578.0, 603.0, 606.0, 618.0, 632.0, 641.0, 651.0, 665.0, 674.0, 682.0, 697.0, 717.0, 746.0, 747.0, 748.0, 753.0, 778.0, 788.0, 792.0, 806.0, 812.0, 816.0, 832.0, 838.0, 839.0, 865.0, 872.0, 882.0, 887.0, 892.0, 902.0, 917.0, 921.0, 936.0, 942.0, 946.0, 947.0, 967.0, 968.0, 971.0, 978.0, 1000.0, 1003.0, 1008.0, 1031.0, 1035.0, 1042.0, 1064.0, 1078.0, 1082.0, 1117.0, 1122.0, 1132.0, 1143.0, 1154.0, 1166.0, 1171.0, 1182.0, 1183.0, 1185.0, 1199.0, 1208.0, 1214.0, 1221.0, 1227.0, 1231.0, 1241.0, 1243.0, 1249.0, 1287.0, 1298.0, 1302.0, 1319.0, 1325.0, 1339.0, 1344.0, 1356.0, 1361.0, 1379.0, 1386.0, 1389.0, 1426.0, 1445.0, 1452.0, 1468.0, 1480.0, 1491.0, 1515.0, 1516.0, 1521.0, 1537.0, 1561.0, 1564.0, 1608.0, 1631.0, 1635.0, 1640.0, 1666.0, 1678.0, 1687.0, 1689.0, 1717.0, 1756.0, 2002.0, 3164.0, 3165.0, 3185.0, 3192.0, 3193.0, 3194.0, 3197.0, 3198.0, 3203.0, 3204.0, 3205.0, 3207.0, 3207.0, 3214.0, 3219.0, 3220.0, 3220.5, 3224.0, 3235.0, 3236.0]

freq_num = np.array(freqMol2_LE)

coup_num_LE = coup_num_LE * freq_num / np.sqrt(2)  # + list([1.15*x for x in back_coup])
coup_num_CT = coup_num_CT * freq_num / np.sqrt(2)  # + list([-1.15*x for x in back_coup])

coup_mat = [np.diag([x, y]) for x, y in zip(coup_num_LE, coup_num_CT)]
reorg1 = sum([(coup_num_LE[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
reorg2 = sum([(coup_num_CT[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
print("Reorg", reorg1, reorg2)
print(f"Len {len(coup_mat)}")

temp = 300
eth = SpinBoson(pd, coup_mat=coup_mat, freq=freq_num, temp=temp)
np.set_printoptions(suppress=True)

coup = eth.coup_mat_np[:,0,0]
freq = eth.freq

# coup = [1,1.5]*10
# freq = [2,10]*10
print(f"Original Freq: {freq}")
tri, Q = lanczos(A=np.diag(freq), p=coup)
res = np.diagonal(Q.T@Q-np.eye(Q.shape[0]))
res = res@res
print(res)

eig = sp.linalg.eigvals(tri)
eig.sort()
print(f"Lanczos Eig {eig}")
print(f"diff {list(eig - freq)}")
print(f"Residual: {res}")

