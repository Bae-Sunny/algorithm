<template>
    <div id="content" class="center-content">
        <h2 class="text-center wow fadeInUp" data-wow-duration="1s" data-wow-delay="0.5s">커피 원두 등급 예측</h2>
        <div class="row justify-content-center">
            <form @submit.prevent="submit" class="wow fadeIn" data-wow-duration="1s" data-wow-delay="0.9s">
                <fieldset>
                    <div class="form-group">
                        <label for="Aroma">아로마</label>
                        <input type="number" class="form-control" v-model="Aroma" step="0.01" min="0" max="10" required>
                    </div>
                    <div class="form-group">
                        <label for="Flavor">맛</label>
                        <input type="number" class="form-control" v-model="Flavor" step="0.01" min="0" max="10"
                            required>
                    </div>
                    <div class="form-group">
                        <label for="Aftertaste">끝맛</label>
                        <input type="number" class="form-control" v-model="Aftertaste" step="0.01" min="0" max="10"
                            required>
                    </div>
                    <div class="form-group">
                        <label for="Acidity">산미</label>
                        <input type="number" class="form-control" v-model="Acidity" step="0.01" min="0" max="10"
                            required>
                    </div>
                    <div class="form-group">
                        <label for="Body">바디감</label>
                        <input type="number" class="form-control" v-model="Body" step="0.01" min="0" max="10" required>
                    </div>
                    <div class="form-group">
                        <label for="Balance">밸런스</label>
                        <input type="number" class="form-control" v-model="Balance" step="0.01" min="0" max="10"
                            required>
                    </div>
                    <div class="form-group">
                        <label for="Uniformity">균일성</label>
                        <input type="number" class="form-control" v-model="Uniformity" step="0.01" min="0" max="10"
                            required>
                    </div>
                    <div class="form-group">
                        <label for="Moisture">수분함량</label>
                        <input type="number" class="form-control" v-model="Moisture" step="0.01" min="0" max="100"
                            required>
                    </div>
                    <div class="form-group">
                        <label for="Altitude_Mean_Meters">재배고도</label>
                        <input type="number" class="form-control" v-model="Altitude_Mean_Meters" step="0.01" min="0"
                            max="3000" required>
                    </div>
                    <button type="submit" class="subbtn">원두 등급 예측</button>
                </fieldset>
            </form>
        </div>
    </div>
</template>

<script>
import axios from 'axios';

export default {
    name: 'CoffeeFormView',
    data() {
        return {
            Aroma: "",
            Flavor: "",
            Aftertaste: "",
            Acidity: "",
            Body: "",
            Balance: "",
            Uniformity: "",
            Moisture: "",
            Altitude_Mean_Meters: "",
        }
    },
    methods: {
        submit() {
            // 데이터를 Django로 전송
            const CoffeeData = {
                Aroma: this.Aroma,
                Flavor: this.Flavor,
                Aftertaste: this.Aftertaste,
                Acidity: this.Acidity,
                Body: this.Body,
                Balance: this.Balance,
                Uniformity: this.Uniformity,
                Moisture: this.Moisture,
                Altitude_Mean_Meters: this.Altitude_Mean_Meters,
            };
            axios.post('http://192.168.0.28:9000/coffeeProject/coffeePredict', CoffeeData)
                .then((resp) => {
                    console.log('Django에서 예측된 원두 등급:', resp.data);
                    // // 결과 페이지로 이동, query string으로 데이터 전달
                    this.$router.push({ name: 'ResultView', query: { pred: resp.data.pred } });
                })
                .catch((err) => {
                    console.log(err) // 에러가 발생하면 처리
                });

        }
    }
}
</script>

<style scoped>
.center-content {
    margin: 20px auto;
    padding: 20px;
    border: 1px solid #ccc;
    border-radius: 5px;
    background-color: #f9f9f9;
    width: 1024px;
}

form {
    display: flex;
    flex-direction: column;
    width: 50%;
}

label {
    margin: 10px 10px;
    text-align: left;
    font-weight: 700;
    font-size: 25px;
}

input {
    background-color: #f4eeed;
    color: black;
    margin: 5px 20px;
    padding: 15px;
    border-radius: 5px;
    font-weight: 700;
    font-size: 20px;
    width: 94%;
}

.subbtn {
    background-color: #60403b;
    color: white;
    margin: 20px 20px;
    padding: 15px;
    border-radius: 5px;
    font-weight: 700;
    font-size: 25px;
}

.subbtn:hover {
    background-color: black;
    cursor: pointer;
    color: white;
}
</style>
