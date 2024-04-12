import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly
import plotly.express as px
import plotly.graph_objects as go

import lightgbm as lgb
import shap
import os
import pickle

# pyplotを使用する際に注記が出ないようにする文
st.set_option("deprecation.showPyplotGlobalUse", False)

# 関数化する
def main():
    # タイトル
    st.title("機械学習による米国住宅価格の予測")
    st.write("最終更新日: 2024/4/12")

    # サイドバーのmenu
    menu = ["分析概要", "分析目的", "予測結果", "住宅価格の決定要因", "結論"]
    # サイドバーの作成
    chosen_menu = st.sidebar.selectbox(
        "menu選択", menu
    )

    # ファイルの設定
    # 訓練済みのモデルファイル
    model_file = "./lgb_model_3.pkl"
    # 読み込めているかを確認
    is_model_file = os.path.isfile(model_file)

    # 前処理前のtrain_data
    train_file = "./train.csv"
    # 前処理前のtest_data
    test_file = "./test.csv"
    # 前処理済みの説明変数のtraindata
    x_train_file = "./X_train.csv"
    # 分析過程で作成した検証データの予測と正解データが記載されてcsvfile
    pred_true_file = "./pred_true.csv"

    # 読み込めているかを確認
    is_train_file = os.path.isfile(train_file)
    is_test_file = os.path.isfile(test_file)
    is_x_train_file = os.path.isfile(x_train_file)
    is_pred_true_file = os.path.isfile(pred_true_file)

    # モデルを再学習するかどうか
    # 再学習しないことを宣言
    no_update = True

    # printで出力すると、ターミナルに出る
    # st.writeだとブラウザ上に出る
    print(is_train_file)
    print(is_test_file)
    print(is_x_train_file)
    print(is_pred_true_file)
    print(no_update)
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    X_train = pd.read_csv(x_train_file)
    df_temp = pd.read_csv(pred_true_file)
    print("データを読み込みました")


    lgb_model = pickle.load(open(model_file, 'rb'))
    print("モデルが読み込まれました")
    

    # menuの中身
    # 分析の概要
    if chosen_menu == "分析概要":
        st.subheader("分析概要")
        st.write(" 2016年米国住宅販売データに基づいて、新しい住宅の価格を予測する機械学習モデルモデルを作成しました。")
        st.subheader("データセットの内容")
        st.write("訓練データには、1460件の住宅データ (土地面積・路地へのアクセス・ガレージの面積・住宅価格など)が含まれます。")
        st.write("テストデータには、1459件の住宅データ (住宅価格以外)が含まれます。")
        st.write("(データの詳細な説明は、GitHub上のdata_description.txt（メタデータ）ファイルに記載されています。)")
        st.write("訓練データを用いてモデルを構築して、テストデータの住宅価格(SalePrice)を予測しました。")
        st.write(" ")
        st.write(" ")
        st.text("訓練データ")
        st.dataframe(train_df.head())
        st.write(" ")
        st.text("テストデータ")
        st.dataframe(test_df.head())


    # データセットの概要
    elif chosen_menu == "分析目的":
        st.subheader("分析目的")
        st.write("・住宅住宅の属性データ(土地面積・路地へのアクセス・ガレージの面積など)を基に、住宅価格を予測すること")
        st.write("・住宅価格に影響を与えた要因を分析すること")

    elif chosen_menu == "予測結果":
        st.subheader("予測結果")
        fig = go.Figure()

        # 予測結果の可視化
        fig.add_trace(go.Scatter(x=df_temp.index, y=df_temp["prediction_values"],
                            mode='markers',
                            name='予測値',
                            marker={"color":"blue"}))
        # 実際のデータ
        fig.add_trace(go.Scatter(x=df_temp.index, y=df_temp["true_values"],
                            mode='lines',
                            name='実測値',
                            marker={"color":"red"}))
        # グラフの設定
        fig.update_layout(title='予測値と実測値の比較',
                        xaxis_title='データポイントのインデックス',
                        yaxis_title='住宅価格(Sales)')
        # streamlitで表示
        st.plotly_chart(fig)


        # 結果についての説明
        st.write("上記の図から構築したモデルが実データに基づいて住宅価格をある程度正確に予測できていることが示唆されています。")
        st.write("検証データに対するRSMEは0.0552であり、これは比較的高い精度で予測できたことを示唆しています。")
        st.write("[RSMEについて]")
        st.write("こ予測と実際の値の誤差の大きさを表し、一般的に値が小さいほど、予測精度が高いことを示します。")

    elif chosen_menu == "住宅価格の決定要因":
        st.subheader("SHAP分析の結果")
        st.write("SHAP値は、各特徴量が予測結果に与える影響度を数値で表す指標です。図表において、中央線が0を表しており、左側の値はプラスの影響、右側の値はマイナスの影響を表します。")

        # shap分析の結果を表示
        st.write("SHAP値は、各特徴量が予測に与える影響度を表す指標です。図では、中央線が0を表しており、左側がプラスの影響、右側がマイナスの影響を表します。")
        st.write("例えば、TotalSF(住宅の総面積)が大きいほど住宅価格は高くなります。")
        st.write("また、日本ではあまり一般的ではありませんが、ScreenPorch（虫よけのガラスで囲まれたテラス）の面積が大きいほど、住宅価格は高くなる傾向があります。")
        st.write("一方で、BsmtUnfSF(未完成の地下室の面積)が大きければ住宅価格は低くなります。")
        st.write(" ")
        st.write("[注意]")
        st.write("SHAP値は、定量的な属性情報の数字的な強さを表すことができますが、定性的な属性情報の数字的な強さを表すことはできません。")
        st.write("例えば、OverallQual(家の素材と仕上げ具合)は定性的な属性情報であり、FeatureValueから住宅価格予測において重要な役割を果たしたことが確かです。")
        st.write("しかし、OverallQualのSHAP値が高いからといって、住宅価格が高いとは必ずしも言えません。")
        # スコアが良いモデルのshap値を求める
        explainer = shap.TreeExplainer(model=lgb_model, data=X_train)
        # shapの値
        shap_values = explainer.shap_values(X=X_train, check_additivity=False)
        fig = shap.summary_plot(shap_values, X_train)
        st.pyplot(fig)

    elif chosen_menu == "結論":
        st.subheader("結論")
        st.write("・検証データに対するRSMEは0.0552であり、未知のデータに対して比較的高い精度で予測できる機械学習モデルが構築できました。")
        st.write("・OverallQual(家の素材と仕上げ具合)やFirePlace(暖炉)の有無が住宅の価格に大きな影響を与えていることがわかりました。")
        st.write("・ScreenPorch（虫よけのガラスで囲まれたテラス）の面積が大きいほど、住宅価格は高くなる傾向があることがわかりました。")


# streamlitを実行したときにmain()を実行するという表記
if __name__ == "__main__":
    main()
