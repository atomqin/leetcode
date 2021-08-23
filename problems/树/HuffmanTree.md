小灰公众号
```java
public class HuffmanTree {
    public static void main(String[] args) {
        int[] weights = new int[] { 2, 3, 7, 9, 18, 25 };
        char[] password = { 'A', 'B', 'C', 'D', 'E', 'F' };
        HuffmanTree hTree = new HuffmanTree();
        hTree.createHuffman(weights);
        hTree.encode(hTree.root, "");
        for (int i = 0; i < password.length; i++) {
            System.out.println(password[i] + ":" + hTree.convertHuffmanToCode(i));
        }
        // hTree.output(hTree.root);
    }
    //树根结点
    private Node root;
    //叶子结点列表
    private Node[] nodes;

    public void createHuffman(int[] weights) {
        Queue<Node> nodeQueue = new PriorityQueue<>();
        nodes = new Node[weights.length];
        for (int i = 0; i < weights.length; i++) {
            nodes[i] = new Node(weights[i]);
            nodeQueue.add(nodes[i]);
        }
        while (nodeQueue.size() > 1) {
            Node lChild = nodeQueue.poll();
            Node rChild = nodeQueue.poll();
            Node parent = new Node(lChild.weight + rChild.weight, lChild, rChild);
            nodeQueue.add(parent);
        }
        root = nodeQueue.poll();
    }

    public void encode(Node root, String code) {
        if (root == null)
            return;
        root.code = code;
        encode(root.lChild, root.code + "0");
        encode(root.rChild, root.code + "1");

    }

    public String convertHuffmanToCode(int index) {
        return nodes[index].code;
    }
    public void output(Node head) {
        if (head == null)
            return;
        System.out.println(head.weight);
        output(head.lChild);
        output(head.rChild);
    }
    class Node implements Comparable<Node>{
        int weight;
        Node lChild;
        Node rChild;
        String code;
        public Node(int weight) {
            this.weight = weight;
        }

        public Node(int weight, Node left, Node right) {
            this.weight = weight;
            this.lChild = left;
            this.rChild = right;
        }
        
        public int compareTo(Node o) {
            return this.weight - o.weight;
        }
    }
}
```
