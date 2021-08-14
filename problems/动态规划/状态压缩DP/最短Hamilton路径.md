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
