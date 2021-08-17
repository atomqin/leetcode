**题目描述**：输入一个数 n，代表编号 `0~n-1` 的 n 个点，求**不重不漏**地从起点 0 走到终点 n - 1 的所有路径中最短的路径长度
```
输入格式
第一行输入整数 n。

接下来 n 行每行 n 个整数，其中第 i 行第 j 个整数表示点 i 到 j 的距离（记为 a[i,j]）。

对于任意的 x,y,z，数据保证 a[x,x]=0，a[x,y]=a[y,x] 并且 a[x,y]+a[y,z]≥a[x,z]。

输出格式
输出一个整数，表示最短 Hamilton 路径的长度。

数据范围
1≤n≤20
0≤a[i,j]≤107
输入样例：
5
0 2 4 5 1
2 0 6 5 3
4 6 0 8 3
5 5 8 0 5
1 3 3 5 0
输出样例：
18
```
```
i:转态i中将经过的点标为1，如101表示经过了点0和点2
j:终点
要求的是dp[(1 << n) - 1][n - 1]，将所有点都标1,状态总数为 1 << n，最终状态即为 (1 << n) - 1
初始状态dp[1][0]=1

通过枚举最后第二个经过的点k来写表达式dp[i][j] = Math.min(dp[i][j], dp[i - {j}][k] + w[k][j]); 其中 i - {j} 相当于 i - (1 << j)
```
```java
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[][] w = new int[20][20];
        int[][] dp = new int[1 << 20][20];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                w[i][j] = sc.nextInt();
            }
        }
        for(int[] d : dp)
            Arrays.fill(d, 0x3f3f3f3f);
        dp[1][0] = 0;
        //优先级排序：(+-) > (<<>>) > (==) > (&) 单目 > 算术 > 位移 > 关系 > 逻辑 > 三目 > 赋值
        for (int i = 0; i < 1 << n; i++) {
            for (int j = 0; j < n; j++) {
                if ((i >> j & 1) == 1) {
                    for (int k = 0; k < n; k++) {
                        if(((i - (1 << j)) >> k & 1) == 1)
                            dp[i][j] = Math.min(dp[i][j], dp[i - (1 << j)][k] + w[k][j]);
                    }
                }
            }
        }
        System.out.println(dp[(1 << n) - 1][n - 1]);
    }
}
```
