import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import pyecharts.options as opts
from pyecharts.charts import Bar
import numpy as np
from pre_process import preprocess_raw_data
from csp_fbcsp import apply_csp
from tqdm import tqdm

def load_data(session, total):
    x_data, y_data = [], []
    root_path = f'input/sess{session:02d}/sess{session:02d}'
    for i in tqdm(range(1, total + 1), desc='Load data'):
        base_path = f'_subj{i:02}_EEG_MI.mat'
        path = root_path + base_path
        # raw 数据中的 'train'标签和'test'标签数据
        x_train, y_train = preprocess_raw_data(path, 'train')
        x_test, y_test = preprocess_raw_data(path, 'test')
        x_data.append(np.concatenate([x_train, x_test]))
        y_data.append(np.concatenate([y_train, y_test]))

    return np.array(x_data), np.array(y_data)


def evaluate_models(x_data_sess01, y_data_sess01):
    loo = LeaveOneOut()
    models = {
        'SVM': svm.SVC(C=10, kernel='linear', probability=True),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(n_neighbors=6),
        'GaussianNB': GaussianNB(),
        'RandomForest': RandomForestClassifier()
    }
    accuracy_results = {name: [] for name in models}
    for name, model in models.items():
        loo_acc = []
        for train_idx, test_idx in tqdm(loo.split(x_data_sess01), desc='Training...'):
            X_train, X_test = x_data_sess01[train_idx], x_data_sess01[test_idx]
            y_train, y_test = y_data_sess01[train_idx], y_data_sess01[test_idx]
            
            X_train = np.concatenate(X_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)
            X_test = X_test[0]
            y_test = y_test[0]

            # 1.使用 csp 进行特征提取
            X_train, csp = apply_csp(X_train, y_train)
            X_test = csp.transform(X_test)
            # 2.进行归一化操作
            # scaler = StandardScaler()
            # X_train_norm = scaler.fit_transform(X_train)
            # X_test_norm = scaler.transform(X_test)
            
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            loo_acc.append(acc)    
        accuracy_results[name] = loo_acc

    return accuracy_results


def print_results(accuracy_results):
    for name, accuracies in accuracy_results.items():
        print(f'{name} 模型: 平均得分={statistics.mean(accuracies):.2f}, 标准差={statistics.stdev(accuracies):.2f}')


def visualize_results(accuracy_results, total):
    x_labels = [f'受试者-{i + 1}' for i in range(total)]
    bar = (
        Bar(init_opts=opts.InitOpts(width="1600px"))
        .add_xaxis(xaxis_data=x_labels)
        .set_global_opts(title_opts=opts.TitleOpts(title="模型准确率对比"))
    )
    for name, accuracies in accuracy_results.items():
        bar.add_yaxis(series_name=name, y_axis=accuracies)
    bar.render("cross_subject_model_accuracy_comparison.html")


def main():
    print('Loading data...')
    X_train, y_train = load_data(1, 54)
    print('Evaluate...')
    accuracy_results = evaluate_models(X_train, y_train)
    print_results(accuracy_results)
    print('Visualize...')
    visualize_results(accuracy_results, 54)


if __name__ == "__main__":
    print('Start...')
    main()
    print('Done!')