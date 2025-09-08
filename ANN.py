import argparse, csv, json, math, os, random, sys, time

# 初始化矩阵和向量
def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def randn(rows, cols, rng, scale=0.1):
    m = []
    for _ in range(rows):
        row = []
        for __ in range(cols):
            u1 = max(1e-12, rng.random())
            u2 = rng.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
            row.append(z * scale)
        m.append(row)
    return m

def vec_add(a, b): return [x + y for x, y in zip(a, b)]
def vec_sub(a, b): return [x - y for x, y in zip(a, b)]
def vec_mul(a, b): return [x * y for x, y in zip(a, b)]
def vec_scalar(a, s): return [x * s for x in a]

def mat_vec_mul(W, v):
    return [sum(wi * vi for wi, vi in zip(row, v)) for row in W]

def outer(u, v): return [[ui * vj for vj in v] for ui in u]
def mat_t(W): return list(map(list, zip(*W)))

# 激活函数和导数
def sigmoid(v):
    out = []
    for x in v:
        if x >= 0:
            z = math.exp(-x)
            out.append(1 / (1 + z))
        else:
            z = math.exp(x)
            out.append(z / (1 + z))
    return out

def sigmoid_deriv(y): return [yi * (1 - yi) for yi in y]

# 定义TinyMLP类
class TinyMLP:
    def __init__(self, in_dim, hidden_dim, out_dim, lr=0.1, rng=None):
        self.in_dim, self.hidden_dim, self.out_dim = in_dim, hidden_dim, out_dim
        self.lr = lr
        self.rng = rng or random.Random()
        self.Wih = randn(hidden_dim, in_dim, self.rng, scale=1 / math.sqrt(in_dim))
        self.bh = [0.0] * hidden_dim
        self.Who = randn(out_dim, hidden_dim, self.rng, scale=1 / math.sqrt(hidden_dim))
        self.bo = [0.0] * out_dim

    def forward(self, x):
        h_in = vec_add(mat_vec_mul(self.Wih, x), self.bh)
        h = sigmoid(h_in)
        o_in = vec_add(mat_vec_mul(self.Who, h), self.bo)
        y = sigmoid(o_in)
        return y, (x, h, y)

    def train_one(self, x, t):
        y, (x0, h, y0) = self.forward(x)
        e_o = vec_sub(t, y)
        g_o = vec_mul(e_o, sigmoid_deriv(y))
        g_h = vec_mul(mat_vec_mul(mat_t(self.Who), g_o), sigmoid_deriv(h))
        dWho = outer(g_o, h)
        dWih = outer(g_h, x0)

        # 更新权重
        for i in range(self.out_dim):
            for j in range(self.hidden_dim):
                self.Who[i][j] += self.lr * dWho[i][j]
        for i in range(self.hidden_dim):
            for j in range(self.in_dim):
                self.Wih[i][j] += self.lr * dWih[i][j]

        self.bo = vec_add(self.bo, vec_scalar(g_o, self.lr))
        self.bh = vec_add(self.bh, vec_scalar(g_h, self.lr))

        return sum(e * e for e in e_o)

    def predict(self, x):
        y, _ = self.forward(x)
        return max(enumerate(y), key=lambda a: a[1])[0], y

    def to_dict(self):
        return {
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "lr": self.lr,
            "Wih": self.Wih,
            "bh": self.bh,
            "Who": self.Who,
            "bo": self.bo,
        }

    @classmethod
    def from_dict(cls, d):
        net = cls(d["in_dim"], d["hidden_dim"], d["out_dim"], d.get("lr", 0.1))
        net.Wih, net.bh = d["Wih"], d["bh"]
        net.Who, net.bo = d["Who"], d["bo"]
        return net

# 预处理像素数据
def scale_pixel(v): return (v / 255.0) * 0.99 + 0.01
def one_hot(label, size=10): return [0.99 if i == label else 0.01 for i in range(size)]

# 读取CSV文件
def read_csv_lines(path, limit=None):
    data = []
    with open(path, "r") as f:
        rdr = csv.reader(f)
        for i, row in enumerate(rdr):
            if limit and i >= limit: break
            label = int(row[0])
            pixels = [scale_pixel(float(x)) for x in row[1:785]]
            data.append((label, pixels))
    return data

# 查找数据文件
def find_data_files(data_arg, want_train=True):
    if os.path.isdir(data_arg):
        return os.path.join(data_arg, "mnist_train.csv" if want_train else "mnist_test.csv")
    return data_arg

# 计算准确率
def accuracy(model, dataset):
    return sum(model.predict(x)[0] == lbl for lbl, x in dataset) / len(dataset)

# 主函数
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hidden", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train-limit", type=int)
    ap.add_argument("--test-limit", type=int)
    ap.add_argument("--save", type=str)
    ap.add_argument("--load", type=str)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train_path = find_data_files(args.data, True)
    test_path = find_data_files(args.data, False)

    train_data = read_csv_lines(train_path, args.train_limit)
    test_data = read_csv_lines(test_path, args.test_limit) if os.path.exists(test_path) else []

    if args.load and os.path.exists(args.load):
        with open(args.load, "r") as f:
            net = TinyMLP.from_dict(json.load(f))
            net.lr = args.lr
    else:
        net = TinyMLP(784, args.hidden, 10, lr=args.lr, rng=rng)

    if args.epochs > 0:
        print(f"[INFO] Training {len(train_data)} samples for {args.epochs} epochs")
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            rng.shuffle(train_data)
            se_sum = 0
            for lbl, x in train_data:
                tvec = one_hot(lbl, 10)
                se_sum += net.train_one(x, tvec)
            acc = accuracy(net, train_data[:min(1000, len(train_data))])
            print(f"Epoch {ep} | MSE={se_sum/len(train_data):.5f} | Train acc≈{acc*100:.2f}% | Time={time.time()-t0:.1f}s")

    if test_data:
        acc = accuracy(net, test_data)
        print(f"[INFO] Test accuracy: {acc*100:.2f}% on {len(test_data)} samples")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(net.to_dict(), f)
        print(f"[INFO] Weights saved to {args.save}")

if __name__ == "__main__":
    sys.argv = [
        "ANN.py",
        "--data", "data",           # 自动读取 data/mnist_train.csv 和 test
        "--hidden", "100",          # 隐藏层大小
        "--epochs", "5",            # 训练轮数
        "--lr", "0.1",              # 学习率
        "--train-limit", "100",     # 加快调试
        "--test-limit", "20",
        "--save", "data/weights.json"  # 保存路径
    ]
    main()
