```
考虑状态设置为最长公共子序列和最长上升子序列的状态”合并”在一起的状态。

设 f[i][j] 为 a 前 i 个和 b 前 j 个匹配的最长公共上升子序列以 b[j] 结尾的长度。（其实这里设置成 a[i] 结尾也可以，只不过为了后面好转移设的是 b[j]）

转移方程就很好推了:

f[i][j]=max{f[i−1][j]  (a[i]≠b[j])}
f[i][j]=max{f[i−1][k] (1≤k<j&&b[k]<b[j])}+1  (a[i]==b[j])
```
![](https://cdn.acwing.com/media/article/image/2020/07/09/37963_97583cd2c1-123.png)
- 写法一

时间复杂度O(n^3)
```java
import java.util.Scanner;

public class Main {
    static int N = 3010;
    static int[] a = new int[N];
    static int[] b = new int[N];
    static int[][] f = new int[N][N];
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        int n = scan.nextInt();
        for(int i = 1;i <= n;i ++) a[i] = scan.nextInt();
        for(int i = 1;i <= n;i ++) b[i] = scan.nextInt();
        for(int i = 1;i <= n;i ++)
        {
            for(int j = 1;j <= n;j ++)
            {
                //a[i] != b[j]
                f[i][j] = f[i - 1][j];
                if(a[i] == b[j])
                {
                    //LCIS中的b序列倒数第二个可能为空
                    f[i][j] = Math.max(f[i][j], 1)
                    //其实就是初始化为1，可以直接这么写
                    //f[i][j] = 1;
                    for(int k = 1;k < j;k ++)
                    {
                        if(b[k] < b[j]) f[i][j] = Math.max(f[i][j], f[i - 1][k] + 1);
                    }
                }
            }
        }
        //需要类似最长上升子序列求得最大值,也可以直接加到上面的循环中
        int res = 0;
        for(int i = 1;i <= n;i ++) res = Math.max(res,f[n][i]);
        System.out.println(res);
    }
}
```
`if (a[i] == b[j])`那段，可以换成下面这么写
```java
int maxv = 1;
for(int k = 1;k < j;k ++)
{
    if (b[k] < b[j])
        maxv = Math.max(maxv, f[i - 1][k] + 1);
}
f[i][j] = Math.max(f[i][j], maxv);
```
- 优化

时间复杂度O(n^2)

[题解](https://www.acwing.com/solution/content/15870/)
```
可以发现a[i] == b[j]情况下if (b[j] > b[k])可等同于if (a[i] > b[k])，所代表意义为求满足a[i] > b[k]情况下所有的f[i - 1][k] + 1的最大值,无需遍历k，可在前两个循环中求出
我们可以维护一个MAX值来储存最大的f[i-1][k]值。即只要有a[i] > a[j]的地方，那么我们就可以更新最大值，所以，当a[i] == b[j]的时候，f[i][j] = MAX+1
```
```java
import java.util.Scanner;

public class Main {
    static int N = 3010;
    static int[] a = new int[N];
    static int[] b = new int[N];
    static int[][] f = new int[N][N];
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        int n = scan.nextInt();
        for(int i = 1;i <= n;i ++) a[i] = scan.nextInt();
        for(int i = 1;i <= n;i ++) b[i] = scan.nextInt();
        for(int i = 1;i <= n;i ++)
        {
            int maxv = 1;//记录当前a[i] > 所有b[k]时，f[k] + 1的最大值，其中k < j
            for(int j = 1;j <= n;j ++)
            {
                f[i][j] = f[i - 1][j];
                if(a[i] == b[j]) f[i][j] = Math.max(f[i][j], maxv);
                if(b[j] < a[i]) maxv = Math.max(maxv, f[i - 1][j] + 1);
            }
            
        }
        //直接这么写更好理解
        /*
        for (int i = 1; i <= n; i++) {
            //注意是0,不然 a[i]==b[j] 时，会更新成 2
            int maxv = 0;
            for (int j = 1; j <= n; j++) {
                f[i][j] = f[i - 1][j];
                if(b[j] < a[i])
                    maxv = Math.max(maxv, f[i - 1][j]);
                if (a[i] == b[j])
                    f[i][j] = maxv + 1;
            }
        }
        */
        //需要类似最长上升子序列求得最大值,也可以直接加到上面的循环中
        int res = 0;
        for(int i = 1;i <= n;i ++) res = Math.max(res,f[n][i]);
        System.out.println(res);
    }
}
```
- 优化成一维

```java
public class Main {
    static int N = 3010;
    static int[] a = new int[N];
    static int[] b = new int[N];
    static int[] f = new int[N];
    static Scanner sc = new Scanner(System.in);
    static PrintWriter writer = new PrintWriter(System.out);
    public static void main(String[] args){
        int n = sc.nextInt();
        for (int i = 1; i <= n; i++) {
            a[i] = sc.nextInt();
        }
        for (int i = 1; i <= n; i++) {
            b[i] = sc.nextInt();
        }
        for (int i = 1; i <= n; i++) {
            int maxv = 0;
            for (int j = 1; j <= n; j++) {
                // f[i][j] = f[i - 1][j];
                if(b[j] < a[i])
                    maxv = Math.max(maxv, f[j]);
                if (a[i] == b[j])
                    f[j] = maxv + 1;
            }
        }
        int res = 0;
        for (int i = 1; i <= n; i++) {
            res = Math.max(res, f[i]);
        }
        writer.println(res);
        writer.flush();
        writer.close();
        
    }
}
```
