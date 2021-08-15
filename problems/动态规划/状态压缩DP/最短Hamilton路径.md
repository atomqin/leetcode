**题目描述**：输入一个数 n，代表编号 `0~n-1` 的 n 个点，求**不重不漏**地从起点 0 走到终点 n - 1 的所有路径中最短的路径长度
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
        //优先级排序：(+-) > (<<>>) > (==) > (&) 
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
