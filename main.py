

    #region IMPORTs
import os
from Model import model as md
from Api import STT as st
    #endregion



    #region MAIN_ARGs
if __name__ == "__main__": 
    print("보이스피싱 감지 모델 불러오는 중...")
    bt = md.Bert(0,"model_save3.pt")
    print("완료")
    st.checkfile()
    print("지금부터 실시간으로 보이스 피싱 확률을 측정 합니다.")
    while 1:
        st.listen()
        if bt.sentences_predict(st.save()) == 0:
            print("현재 통화는 보이스 피싱일 확률이 적습니다.")
        else:
            print("현재 통화는 보이스 피싱일 확률이 많습니다.")
    #endregion