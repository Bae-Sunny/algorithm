<template>
    <div id="content" class="center-content">
        <h2 class="text-center wow fadeInUp" data-wow-duration="1s" data-wow-delay="0.5s">원두 등급 예측 결과</h2>
        <!-- <p>예측결과: {{ predictedGrade }} 등급 원두</p> -->
        <!-- 이미지 표시 -->
        <div class="result wow fadeIn" data-wow-duration="1s" data-wow-delay="1.5s">
            <img v-if="predictedGrade === '1'" src="img/Grade1.png" alt="grade1">
            <img v-else-if="predictedGrade === '2'" src="img/Grade2.png" alt="grade2">
            <img v-else-if="predictedGrade === '3'" src="img/Grade3.png" alt="grade3">
        </div>
    </div>
</template>

<script>
export default {
    name: 'ResultView',
    computed: {
        predictedGrade() {
            return this.$route.query.pred;
        }
    },
    mounted() {
        console.log('Received prediction:', this.predictedGrade);
    }
};
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

.result {
    display: flex; 
    justify-content: center; 
    padding: 20px;
}

.result img {
    max-width: 100%; /* 이미지가 자신의 원래 크기를 넘어가지 않도록 */
    height: auto; /* 가로 세로 비율 유지 */
}
</style>
