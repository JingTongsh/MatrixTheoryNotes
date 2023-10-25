# 第2章

## 第1节 线性空间

### 定义：线性子空间

设 $V$ 是一个线性空间，数域 $\mathbb{F} = \mathbb{R}$ 或 $\mathbb{F} = \mathbb{C}$

子集 $U\subseteq V \cdots$ 至少有0元素

### 性质

设 $V$ 有限维，$U$ 线性子空间

（1）维度
$$
\dim U \le \dim V
$$

（2）传递性：$U$ 是 $V$ 线性子空间，$V$ 是 $W$ 线性子空间，则 $U$ 是 $W$ 线性子空间

### 判定

定理（判定）$U$是 $V$ 的线性子空间 $\iff$ $\forall \alpha, \beta\in U, \alpha + \beta \in U$ 且 $\forall \lambda\in \mathbb{F} \forall \alpha\in U, \lambda \alpha \in U$，即加法封闭且数乘封闭

### 构造新的线性子空间的方法

（1）设 $V$ 是线性空间，$U_i\ (i\in I)$ 是线性子空间

寻找**包含于** $U_1, U_2$ 的线性子空间：

交 $\bigcap_{i\in I} U_i$ 是包含于 $U_i\ (i\in I)$ 的**最大**线性子空间

（2）设 $V$ 是线性空间，$U_1, U_2$ 是线性子空间

寻找**包含** $U_1, U_2$ 的线性子空间：

$U_1 \cup U_2$ 不是线性子空间

定义 **和** $U_1 + U_2 = \{u_1+u_2: u_1\in U_1, u_2\in U_2\}$，是包含 $U_1, U_2$ 的**最小**线性子空间

（3）设 $V$ 是线性空间，$U_1, U_2, \cdots, U_k$ 是线性子空间

寻找**包含** $U_1, U_2, \cdots, U_k$ 的线性子空间：

**和** $U_1 + U_2 + \cdots + U_k = \sum_{j=1}^{k} U_k = \{\}$

（4）如果 $\alpha_1, \cdots, \alpha_k\in V$，由 $\alpha_1, \cdots, \alpha_k$ **张成**的线性子空间 $\{x_1\alpha_1 + \cdots + x_k\alpha_k: x_i\in\mathbb{F}\} = \mathbb{F}\alpha_1 + \cdots \mathbb{F}\alpha_k$，就是其线性组合的集合

### 维数定理

定理 设 $V$ 是一个有限维线性空间，$U, W$ 是 $V$ 的线性子空间，那么
$$
\dim (U+W) = \dim U + \dim W - \dim (U\cap W)
$$

Pf. 设 $\dim (U\cap W) = l$，有一组基……

### 定义：直和

设 $V$ 是线性空间，$U, W$ 是线性子空间

如果 $U\cap W = \{0\}$，那么 $U+W = U\oplus W$ 称为 $U, W$ 的**直和**

设 $V$ 是线性空间，$U_1, U_2, \cdots, U_k$ 是线性子空间，归纳定义**直和** $U_1 \oplus U_2 \oplus \cdots \oplus U_k$

### 直和的基本性质、等价刻画

设 $V$ 是有限维线性空间，$U, W$ 是线性子空间，则以下条件等价

* $U+W$ 是直和
* $\forall \alpha \in U+W, \alpha = u+w, u\in U, w\in W$ 表达式唯一
* $0=u+w, u\in U, w\in W\implies u=0\land w=0$
* $\dim(U+W) = \dim U + \dim W$

### 定义：补子空间

设 $V$ 是线性空间，$U$ 是 $V$ 的线性子空间，如果存在另一个线性子空间 $W$，使得 $U\oplus W = V$，即

* $U\cap W = \{0\}$
* $U+W=V$

那么 $W$ 称为 $U$ 的**补空间**。

例：$V = \mathbb{R}^3$，$U$ 是 $xy$ 平面，$W$ 是 $z$ 轴

例：$V$ 为次数不超过2的多项式的集合，$V = \{\}$，
$U = \{f\in V : f(0) = 0\}$，求补空间？

解：$U = \{f: f(x) = a_2x^2 + a_1x, a_i\in\mathbb{F}\}$，其补空间 $W = \{f: f = c\in \mathbb{F}\}$ 是常值多项式的集合

### 一个矩阵对应的4个经典的线性子空间

设 $A_{m\times n}$，列向量 $A = (\alpha_1, \cdots, \alpha_n)$，行向量 $A = \left(\begin{array}{c}\beta_1^{\rm T} \\ \vdots \\ \beta_m^{\rm T}\end{array}\right)$

（1）$A$ 的**零空间** $N(A) = \{x | Ax=0\}$

（2）$A$ 的**列空间** $R(A) = Span \{\alpha_1, \cdots, \alpha_n\}$

（3）$A$ 的**行空间** $R(A^{\rm T}) = Span \{\beta_1, \cdots, \beta_n\}$

（4）$A$ 的**左零空间** $N(A^{\rm T}) = \{x | x^{\rm T}A=0\} = \{x | A^{\rm T} x=0\}$

给定一个具体的矩阵 $A$，如何计算这4个线性子空间？

求解齐次线性方程组 $Ax=0$，由基础解系张成零空间；在求解过程中化 $A$ 为行简化阶梯型 $A_1$，其行空间相同；

求解齐次线性方程组 $A^{\rm T}x=0$，由基础解系张成 $A$ 的左零空间；在求解过程中化 $A^{\rm T}$ 为行简化阶梯型 $A_2$，其行空间就是 $A$ 的列空间。

## 第2节 线性变换（线性映射）

### 定义：线性变换（线性映射）

设 $U, V$ 是线性空间，映射 $\sigma: U\to V$，如果 $\sigma$ 保持线性结构，即

* 线性性 $\forall\alpha, \beta\in U, \sigma(\alpha+\beta) = \sigma(\alpha) + \sigma(\beta)$
* 数乘 $\forall k\in\mathbb{F}, \forall\alpha\in U, \sigma(k\alpha) = k\cdot\sigma(\alpha)$

称 $\sigma$ 为从 $U$ 到 $V$ 的**线性变换**或**线性映射**

注：上述两个条件可以合并为一个

$$
\forall k, l\in\mathbb{F}, \forall\alpha, \beta\in U, \sigma(k\alpha + l\beta) = k\sigma(\alpha) + l\sigma(\beta)
$$

推广到任意有限多个向量：设$\sigma$ 为从 $U$ 到 $V$ 的线性映射，$\alpha_1, \cdots, \alpha_n \in U, x_1, \cdots, x_n \in \mathbb{F}$，那么
$$
\sigma(x_1\alpha_1 + \cdots + x_n\alpha_n) = x_1\sigma(\alpha_1) + \cdots x_n\sigma(\alpha_n)
$$

### 基本性质

* $\sigma(0) = 0$
* $\sigma(-\alpha) = -\sigma(\alpha)$

### 一些记号

设 $U, V$ 是线性空间

$Hom(U, V) = \{\sigma | \sigma 是U到V线性映射\}$

$End(U) = Hom(U, U)$

$U^{*} = Hom(U, \mathbb{F})$

### 定义：单、满、同构

设 $\sigma\in Hom (U, V)$，单射称为**单变换**，满射称为**满变换**；

$\sigma$ 既单又满称为**同构映射**，称 $U, V$ **同构**，记为 $U\cong V$

例：闭区间上无穷阶可导光滑函数集合 $V = C^{\infty} [a, b], \sigma: V\to V, f(x)\to f'(x)$，求导是线性映射

例：$V = C [a, b], \sigma: V\to \mathbb{F}=\mathbb{R}, f(x)\to \int_a^b f(x)dx, \sigma\in Hom (V, \mathbb{F})$

### 线性映射的性质

设 $\sigma\in Hom (U, V)$，如果 $\alpha_1, \cdots, \alpha_n \in U$ 线性相关，那么 $\sigma(\alpha_1), \cdots, \sigma(\alpha_n)$ 线性相关

逆否命题：如果 $\sigma(\alpha_1), \cdots, \sigma(\alpha_n)$ 线性无关，那么 $\alpha_1, \cdots, \alpha_n \in U$ 线性无关

### 构造线性映射的方式

设 $U$ 是有限维线性空间，$V$ 是线性空间，令 $\{\alpha_1, \cdots, \alpha_n\}$ 为 $U$ 的一组基，即 $\forall\alpha\in U$ 存在唯一的一组系数 $k_1, \cdots, k_n$ 使得 $k_1\alpha_1 + k_n\alpha_n = \alpha$

现任取 $n$ 个向量 $\beta_1, \cdots, \beta_n\in V$，令 $\beta_i = \sigma(\alpha_i)$，线性扩张 $\sigma(\alpha) =$

4种特殊的线性变换：零变换，恒等变换，位似变换，可逆变换

### 核，像集

设 $\sigma\in Hom (U, V), U, V$ 是有限维线性空间，定义

$\sigma$ 的**核**是0的原像 $ker(\sigma) = \{x\in U | \sigma(x) = 0\}$

$\sigma$ 的**像集**是像的集合（值域） $Im(\sigma) = \{\beta \in V | \exists\alpha\in U, \sigma(\alpha) = \beta\}$

引理 核是 $U$ 的线性子空间；像集是 $V$ 的线性子空间

例 矩阵 $A_{m\times n}$，映射 $\sigma: \mathbb{F}^n \to \mathbb{F}^m, \sigma(x) = Ax$

则 $ker(\sigma) = \{x | Ax = 0\} = N(A), \dim kern(\sigma) = \dim N(A) =$ 基础解系中的向量个数

$Im(\sigma) = \{x_1\alpha_1 + \cdots + x_n\alpha_n | x_i\in\mathbb{F}\} = R(A), \dim Im(\sigma) = \dim R(A) = r(A)$

定理 设 $U, V$ 有限维线性空间，$\sigma\in Hom(U, V)$，

* $\sigma$ 是单变换 $\iff kern(\sigma) = \{0\}$
* $\sigma$ 是满变换 $\iff Im(\sigma) = V$
* 如果 $\sigma\in Hom(U, U) = End(U)$，那么 $\sigma$ 是单变换 $\iff$ $\sigma$ 是满变换 $\iff$ $\sigma$ 是同构

### 可逆映射

定义：$U, V$ 线性空间，$\sigma\in Hom(U, V)$，如果存在 $\tau\in Hom(V, U)$ 使得

* $\forall \alpha \in U, \tau(\sigma(\alpha)) = \alpha$
* $\forall \beta \in V, \sigma(\tau(\beta)) = \beta$

那么称 $\sigma$ 是可逆映射，$\tau$ 是 $\sigma$ 的逆映射，记为 $\tau = \sigma^{-1}$

定理：设 $U, V$ 线性空间，$\sigma\in Hom(U, V)$，那么 $\sigma$ 同构映射 $\iff$ $\sigma$ 可逆映射

### 联系到矩阵

设 $U, V$ 有限维线性空间，$\sigma\in Hom(U, V)$，$\{\alpha_1, \cdots, \alpha_n\}$ 是 $U$ 的一组基，$\{\beta_1, \cdots, \beta_m\}$ 是 $V$ 的一组基，则
$$
\sigma(\alpha_1) = c_{11} \beta_1 + \cdots + c_{1m}\beta_m \\
\vdots \\
\sigma(\alpha_n) = c_{n1} \beta_1 + \cdots + c_{nm}\beta_m
$$
写成分块矩阵形式
$$
(\sigma(\alpha_1) \cdots \sigma(\alpha_n)) = (\beta_1 \cdots \beta_m) \left(\begin{array}{ccc}c_{11} & \cdots & c_{m1} \\ c_{m1} & \cdots & c_{mn}\end{array}\right) = (\beta_1 \cdots \beta_m) A
$$
考虑坐标
$$
x\in U, x = x_1 \alpha_1 + \cdots x_n \alpha_n \\
y\in V, y = y_1 \beta_1 + \cdots y_m \beta_m
$$
坐标变换
$$
\left(\begin{array}{c}y_1 \\ y_m\end{array}\right) = A \left(\begin{array}{c}x_1 \\ x_n\end{array}\right)
$$

设 $\{\alpha_1', \cdots, \alpha_n'\}$ 是 $U$ 的一组基，$\{\beta_1', \cdots, \beta_m'\}$ 是 $V$ 的一组基，则
$$
(\sigma(\alpha_1') \cdots \sigma(\alpha_n')) = (\beta_1' \cdots \beta_m') A'
$$

相同线性空间不同基之间的关系，$P$ 可逆矩阵
$$
(\alpha_1', \cdots, \alpha_n') = (\alpha_1, \cdots, \alpha_n) P \\
(\sigma(\alpha_1') \cdots \sigma(\alpha_n')) = (\beta_1 \cdots \beta_m) A P \\
(\sigma(\alpha_1') \cdots \sigma(\alpha_n')) = (\beta_1' \cdots \beta_m') A' = (\beta_1 \cdots \beta_m) Q A'
$$

因此
$$AP = QA'$$

$\sigma\in End(U) = Hom(U, U)$，$\{\alpha_1, \cdots, \alpha_n\}$ 是 $U$ 的一组基
$$
(\sigma(\alpha_1) \cdots \sigma(\alpha_n)) = (\alpha_1 \cdots \alpha_n) A
$$

则
$$
AP = PA'
$$
因此
$$
A' = P^{-1} A P
$$
可见具有**相似关系**

定义：矩阵，行列式，迹

设 $\sigma\in End(U)$， $U$ 有限维线性空间，$\{\alpha_1, \cdots, \alpha_n\}$ 是 $U$ 的一组基，$A$ 是 $\sigma$ 在这组基下的矩阵

* $\sigma$ 的**行列式** $|\sigma| = |A|$
* $\sigma$ 的**迹** $tr(\sigma) = tr(A)$

定义

$A$ 是 $n\times n$ 方阵，$A^2 = A$，称为**幂等矩阵**；存在 $k\in\mathbb{N}$ 使得 $A^k = O$ 称为**幂零矩阵**

$\sigma = \sigma \circ \sigma$ **幂等映射**；
$\underbrace{\sigma \circ \cdots \circ \sigma}_{k\text{个}} = 0$ **幂零映射**

有如下的等价关系

$$
A^2 = A \iff \sigma \circ \sigma = \sigma \\
A^k = O \iff \sigma^k = 0 \\
$$

定理

设 $\sigma\in End(U)$， $U$ 有限维线性空间，$\{\alpha_1, \cdots, \alpha_n\}$ 是 $U$ 的一组基，$A(\sigma)$ 是 $\sigma$ 在这组基下的矩阵，$\mathbb{F}$ 是数域，由此定义了一个映射
$$
\varphi: End(U) \to M_n(\mathbb{F}), \sigma\mapsto A(\sigma)
$$

有如下结论：

* $End(U)$ 是一个线性空间，$\varphi$ 是从 $End(U)$ 映射到 $M_n(\mathbb{F})$ 的**同构线性映射**
* $\forall \sigma, \tau \in End(U), \varphi(\sigma\circ\tau) = A(\sigma) A(\tau)$

例 $V = \{次数不超过2的多项式\} = \{a_0+a_1t+a_2t^2, a_i\in\mathbb{F}\}, \sigma: V\to V, f\mapsto f'$，一组基 $\{1, t, t^2\}$。

（1）求 $\sigma$ 在这组基下的矩阵；

（2）求行列式和迹；

（3）求 $Im(\sigma), ker(\sigma)$；

（4）$Im(\sigma) + ker(\sigma)$ 是否直和？

解：（1）
$$
(0, 1, 2t) = (1, t, t^2)\left(\begin{array}{ccc}0 & 1 & 0 \\ 0 & 0 & 2 \\ 0 & 0 & 0\end{array}\right)
$$

（2）$0, 0$

（3）
$$
Im(\sigma) = \{次数不超过1的多项式\} = \{a_0+a_1t, a_i\in\mathbb{F}\} \\
ker(\sigma) = \{常数\} = \{a_0, a_0\in\mathbb{F}\}
$$

（4）否

## 第3节 内积空间

设 $V$ 是一个有限维内积空间，$U\subseteq V$ 线性子空间，定义 $U^{\perp} = \{x\in V: x\perp U\} = \{x\in V: \forall u\in U, (x, u) = 0\}$

定理 $V = U\oplus U^{\perp}$

$\mathbb{C}^n$ 内积空间，$x = (x_1, \cdots, x_n)^n, y = (y_1, \cdots, y_n)^n, (x, y) = x_1\overline{y_1} + \cdots + x_n\overline{y_n}$

$U\subseteq V, (U^{\perp})^{\perp} = U$

引理 $A\in\mathbb{C}^{m\times n}$，则

* $N(A) = R(A^*)^{\perp}$
* $R(A) = N(A^*)^{\perp}$

### 常见问题及解法

问题1：最佳逼近问题

$V$ 是有限维欧氏空间，$U\subseteq V$ 是线性子空间，$\forall \beta \in V$ 找 $\alpha \in U$ 使得 $\|\alpha-\beta\|$ 最小

解：设 $\{\alpha_1, \cdots, \alpha_k\}$ 是 $U$ 一组标准正交基，$\alpha = c_1\alpha_1 + \cdots + c_k\alpha_k$

垂直投影
$$
(\beta - \alpha, \alpha_i) = 0, 1\le i\le k\\
\implies (\beta - c_1\alpha_1 - \cdots - c_k\alpha_k, \alpha_i) = 0, 1\le i\le k\\
\implies (\beta - c_i\alpha_i, \alpha_i) = 0\\
\implies c_i = (\beta, \alpha_i)
$$
可见，系数是内积

问题2：$Ax = b$ 最佳近似解，$A\in\mathbb{R}^{m\times n}, b\in\mathbb{R}^{m}$
$$
\min_{x} \quad y = \|Ax - b\|
$$

方法一 令偏导为零，求极值

方法二 转化为上述最佳逼近问题

$b$ 投影到 $R(A) = \{x_1\alpha_1 + \cdots x_n \alpha_n | A = (\alpha_1 \cdots\alpha_n )\} = \{Ax | x\in\mathbb{R}^n\}$

$$
b - Ax_0 \perp R(A)\\
b - Ax_0 \in R(A)^{\perp} = N(A)\\
A^{\top} (Ax_0 - b) = 0
$$

只需求解线性方程组即可。

例：

## 第4节 内积空间中的线性变换

### 定义：保距变换

$V$ 是有限维内积空间，$\sigma\in End(V)$ 保持距离，即 $\forall \alpha, \beta \in V, d(\sigma(\alpha), \sigma(\beta)) = d(\alpha, \beta)$，其中距离定义 $d(\alpha, \beta) = \|\alpha - \beta\|$，称 $\sigma$ 是**保距映射**

### 保距变换的各种等价刻画

**定理** 设 $V$ 是有限维内积空间，$\sigma\in End(V)$，那么以下条件等价：

* $\sigma$ 是保距映射
* $\sigma$ 保持向量长度 $\forall \alpha\in V, \|\sigma(\alpha)\| = \|\alpha\|$
* $\sigma$ 保持内积 $\forall \alpha, \beta\in V, (\sigma(\alpha), \sigma(\beta)) = (\alpha, \beta)$

计算内积的**公式**

设 $V$ 是一个有限维内积空间，$\{\alpha_1, \cdots, \alpha_n\}$ 是 $V$ 的一组标准正交基，
$$
\alpha = \sum x_k\alpha_k, \\
\beta = \sum y_k\alpha_k, \\
$$
则
$$
\begin{aligned}
(\alpha, \beta)
&= \sum_i \sum_j (x_i\alpha_i, y_j\alpha_j) \\
&= \sum_i \sum_j x_i \overline{y_j} (\alpha_i, \alpha_j) \\
&= \sum_i x_i \overline{y_i}.
\end{aligned}
$$

**定理** 设 $V$ 是一个有限维内积空间，$\{\alpha_1, \cdots, \alpha_n\}$ 是 $V$ 的一组标准正交基，$\sigma\in End(V)$，那么 $\sigma$ 是保距变换 $\iff \sigma$ 在 $\{\alpha_1, \cdots, \alpha_n\}$ 基下的矩阵是酉矩阵.

**定理** 设 $V$ 是一个有限维内积空间，$\sigma\in End(V)$，那么 $\sigma$ 是保距变换 $\iff \sigma$ 把一组标准正交基映射到一组标准正交基.

### 例子：二维保距变换

例：$V=\mathbb{R}^2, \sigma: \mathbb{R}^2 \to \mathbb{R}^2, x\mapsto \left(\begin{array}{cc}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{array}\right) x$，

旋转是保距变换.

例：$\tau: \mathbb{R}^2 \to \mathbb{R}^2, (x, y)\mapsto (x, y) \left(\begin{array}{cc}0 & 1\\ -1 & 0\end{array} \right) = (-y, x)$.
基 $e_1 = (1, 0), e_2 = (0, 1)$，$(\tau(e_1), \tau(e_2)) = ((0, 1), (-1, 0)) = (e_1, e_2) \left(\begin{array}{cc}0 & -1\\ 1 & 0\end{array} \right)$
的矩阵是 $\left(\begin{array}{cc}0 & -1\\ 1 & 0\end{array} \right)$ 而不是 $\left(\begin{array}{cc}0 & 1\\ -1 & 0\end{array} \right)$.

严格写出定义式，把基写出来；不看向量，看抽象的元素.

例：二维空间任何一个保距变换是什么样的？
$V=\mathbb{R}^2, \sigma:\mathbb{R}^2 \to \mathbb{R}^2$ 是保距变换.

$\mathbb{R}^2$ 标准正交基 $e_1 = \left(\begin{array}{c}1\\0\end{array}\right), e_2 = \left(\begin{array}{c}0\\1\end{array}\right)$，则 $(\sigma(e_1), \sigma(e_2)) = (e_1, e_2) A.$

$\sigma$ 保距变换 $\iff A$ 酉矩阵 $\iff A^\top A = I_2,$

令 $A = \left(\begin{array}{cc}a & b\\ c & d\end{array} \right)$，则
$$
\begin{cases}
a^2 + c^2 = 1, \\
ab + cd = 0, \\
b^2 + d^2 = 1,
\end{cases}
$$

可见向量 $(a, c)$ 和向量 $(b, d)$ 都是单位向量且相互垂直.

如 $(a, c) = (\cos\theta, \sin\theta)$，则 $(b, d) = (-\sin\theta, \cos\theta)$ 或 $(b, d) = (\sin\theta, -\cos\theta)$，
$A = \left(\begin{array}{cc}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{array}\right)$ 旋转，或
$A = \left(\begin{array}{cc}\cos\theta & \sin\theta\\ \sin\theta & -\cos\theta\end{array}\right)$ 反射（对称）

$V = \mathbb{R}^2$，基 $e_1, e_2$, $\forall x\in V, x = x_1e_1 + x_2e_2$
$$
(\sigma(e_1), \sigma(e_2)) = (e_1, e_2) A
$$

$x$ 坐标 $(x_1, x_2)$；
$\sigma(x)$ 坐标 $A \left(\begin{array}{c}x_1\\ x_2\end{array}\right)$

设 $A = \left(\begin{array}{cc}\cos\theta & \sin\theta\\ \sin\theta & -\cos\theta\end{array}\right)$，特征方程 $\lambda^2 - 1 = 0$，解得 $\lambda = \pm 1$

设 $\lambda_1 = 1$ 对应特征向量 $v_1$，$\lambda_2 = -1$ 对应特征向量 $v_2$，设 $\|v_1\| = \|v_2\| = 1$

$Av_1 = v_1, Av_2 = -v_2$

赫米特矩阵 $A^* = A$ 因此 $v_1\perp v_2$

对称轴 $x_1=x_2$

$l_1 = Span(v_1)$，$A$ 是关于直线 $l_1$ 的反射，$l_1$ 是 $A$ 的对称轴.

一般情况：$V$ 任意二维内积空间，$\sigma: V\to V$ 保距变换，$\{e_1, e_2\}$ 标准正交基
$$
(\sigma(e_1), \sigma(e_2)) = (e_1, e_2) A
$$

称为旋转或反射.

例：$\sigma: \mathbb{R}^2 \to \mathbb{R}^2, x\mapsto Ax$ 保距变换，取 $e_1 = \left(\begin{array}{c}1\\0\end{array}\right), e_2 = \left(\begin{array}{c}0\\1\end{array}\right)$，则
$$
(\sigma(e_1), \sigma(e_2)) = (e_1, e_2) A,
$$
此 $A$ 正是彼 $A$.

### 例子：三维保距变换

例：$V$ 三维内积空间，$\{e_1, e_2, e_3\}$ 标准正交基，$\sigma: V\to V$ 保距变换是什么样的？
$$
(\sigma(e_1), \sigma(e_2), \sigma(e_2)) = (e_1, e_2, e_3) A,
$$
$\sigma$ 保距变换 $\iff A$ 酉矩阵 $\iff A^\top A = I_3 \cdots$

过程复杂，结论如下：

如果 $A^\top A = I_3$，那么 $A$ 只能正交相似于以下6个矩阵：
$$
diag(1, 1, 1), \\
diag(1, 1, -1), \\
diag(1, -1, -1), \\
diag(-1, -1, -1), \\
\left(\begin{array}{ccc}1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta\end{array}\right), \\
\left(\begin{array}{ccc}-1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta\end{array}\right).
$$

其中第5种是绕一个轴的旋转：
如果 $A\sim \left(\begin{array}{ccc}1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta\end{array}\right)$，那么存在 $Q = (\alpha_1, \alpha_2, \alpha_3), Q^\top Q = I_3$，使得
$$
Q^{-1} A Q = \left(\begin{array}{ccc}1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta\end{array}\right)
$$
$A\alpha_1 = \alpha_1, A(\alpha_2, \alpha_3) = (\alpha_2, \alpha_3) \left(\begin{array}{cc}\cos\theta & \sin\theta\\ \sin\theta & -\cos\theta\end{array}\right)$

轴是 $Span(\alpha_1)$，在平面 $Span(\alpha_2, \alpha_3)$ 内旋转.

其中第6种是绕轴 $Span(\alpha_1)$ 的旋转且关于平面 $Span(\alpha_2, \alpha_3)$ 反射.

$\sigma: \mathbb{R}^3 \to \mathbb{R}^3, x\mapsto Ax$ 保距变换，取 $e_1 = \left(\begin{array}{c}1\\0\\0\end{array}\right), e_2 = \left(\begin{array}{c}0\\1\\0\end{array}\right), e_3 = \left(\begin{array}{c}0\\0\\1\end{array}\right)$，则
$$
(\sigma(e_1), \sigma(e_2), \sigma(e_3)) = (e_1, e_2, e_3) A,
$$
此 $A$ 正是彼 $A$.

例：$A = \left(\begin{array}{ccc}\cfrac14 & \cfrac{\sqrt 3}{4} & \cfrac{\sqrt 3}{2} \\ \cfrac{\sqrt 3}{2} & -\cfrac12 & 0 \\ \cfrac{\sqrt 3}{4} & \cfrac34 & -\cfrac12\end{array}\right)$

解：看迹，正交相似不改变迹，不可能是前4种.
第5种有特征值 $1$，第6种有特征值 $-1$.
计算得 $|I-A| = 0,$
$tr(A) = -\cfrac34 = 1 + 2\cos\theta, \cos\theta = -\cfrac78$.

例：设 $v\in \mathbb{C}^n, v\neq 0, H = I_n - 2\cfrac{vv^*}{v^* v}, x\mapsto Hx (x\in\mathbb{C}^n)$，称为 Householder 变换，是保距变换.

$$
Hv = v - 2\cfrac{vv^*v}{v^* v} = v-2v = -v
$$

$U = Span(v)^\perp, x\in U$，那么
$$
Hx = x - 2\cfrac{vv^*x}{v^* v}, \\
v^* x = \sum_k \overline{v_k} x_k = 0, \\
Hx = x
$$

是关于 $U$ 的反射，是保距变换.

例：Jacobi旋转（Givens旋转）……自己看书

### 对称变换

定义：设 $V$ 是一个欧氏空间，即 $\mathbb{F} = \mathbb{R}$ 的内积空间；设 $\sigma\in End(V)$，如果 $\forall \alpha, \beta\in V, (\sigma(\alpha), \beta) = (\alpha, \sigma(\beta))$，那么称 $\sigma$ 是**对称变换**.

定理：设 $V$ 是一个有限维欧氏空间，$\sigma\in End(V)$，那么 $\sigma$ 是对称变换 $\iff \sigma$ 在任意一组标准正交基下的矩阵是对称矩阵.

### 伴随变换

定义：设 $V$ 是一个内积空间，$\mathbb{F} = \mathbb{R}$ 或 $\mathbb{F} = \mathbb{C}$，$\sigma\in End(V)$，如果存在 $\tau\in End(V)$ 使得 $\forall \alpha, \beta\in V, (\sigma(\alpha), \beta) = (\alpha, \tau(\beta))$，那么称 $\tau$ 是 $\sigma$ 的**伴随变换**，记为 $\tau = \sigma^*$.

如果 $\sigma^* = \sigma$，称 $\sigma$ 是**自伴变换**.

定理：设 $V$ 是一个有限维内积空间，$\sigma, \tau \in End(V)$，在某组标准正交基下，其矩阵分别为 $A, B$.
那么：

（1）$(\sigma^*)^* = \sigma, (\sigma+\tau)^* = \sigma^*+\tau^*, (\lambda\sigma)^* = \overline{\lambda}\sigma^*;$

（2）如果 $\tau=\sigma^*$ 那么 $B=A^*$.

定理

$V = U\oplus U^\perp, \forall x\in V$，$x$ 可以分解到 $U$ 和 $U^\perp$ 上.
设 $x$ 在 $U$ 的投影为 $y$，在 $U^\perp$ 的投影为 $z$，$x = y+z$.

定义 $P_U: V\to U\subseteq V, x\mapsto y$，则 $P_U \in End(V)$，称为 $V$ 向子空间 $U$ 作垂直投影的**正交投影变换**；
类似地，$P_{U^\perp}: V\to U^\perp, x\mapsto z$ 为 $V$ 向子空间 $U^\perp$ 作垂直投影的**正交投影变换**.

定理：设 $V$ 内积空间，$\sigma \in End(V)$，那么 $\sigma$ 是 $V$ 向某个子空间的正交投影变换 $\iff \sigma$ 是自伴的幂等变换.
