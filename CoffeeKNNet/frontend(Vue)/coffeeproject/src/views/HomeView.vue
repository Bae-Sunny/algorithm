<template>
  <div class="main-view bg-image">
    <div class="center-content">
      <h1 class="wow fadeInUp" data-wow-duration="1s" data-wow-delay="0.3s">커피 원두 평가</h1>
      <!-- 폼 요소 -->
      <form @submit.prevent="submitForm" class="coffee-form wow fadeIn" data-wow-duration="1s" data-wow-delay="0.5s">
        <div class="form-group">
          <label for="weight">Weight (grams):</label>
          <input type="number" id="weight" v-model="form.weight" required />
        </div>
        <div class="form-group">
          <label for="acidity">Acidity (1-10):</label>
          <input type="number" id="acidity" v-model="form.acidity" min="1" max="10" required />
        </div>
        <div class="form-group">
          <label for="aroma">Aroma (1-10):</label>
          <input type="number" id="aroma" v-model="form.aroma" min="1" max="10" required />
        </div>
        <div class="form-group">
          <label for="flavor">Flavor (1-10):</label>
          <input type="number" id="flavor" v-model="form.flavor" min="1" max="10" required />
        </div>
        <div class="form-group">
          <label for="body">Body (1-10):</label>
          <input type="number" id="body" v-model="form.body" min="1" max="10" required />
        </div>
      </form>

      <!-- 버튼 -->
      <div class="button-container wow fadeIn" data-wow-duration="1s" data-wow-delay="0.9s">
        <button type="submit" class="submit-btn">원두 평가</button>
      </div>

      <!-- CoffeeKNNet 요소 -->
      <h5 class="coffee-knnet wow fadeInUp" data-wow-duration="1s" data-wow-delay="1.6s">
        <span class="coffee">Coffee</span>
        <span class="knn">KNN</span>
        <span class="et">et</span>
      </h5>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      form: {
        beanName: '',
        origin: '',
        weight: null,
        acidity: null,
        aroma: null,
        flavor: null,
        body: null,
      },
    };
  },
  methods: {
    submitForm() {
      // 여기에 폼을 제출하는 로직을 추가할 수 있습니다.
      // 이 예시에서는 submitForm 메서드가 호출되면 폼 데이터를 서버로 전송하거나 처리합니다.
      console.log('폼이 제출되었습니다.');
    },
  },
};
</script>

<style scoped>
.main-view {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 9999;
}

.center-content {
  display: flex;
  flex-direction: column;
  align-items: center; /* 수직 방향으로 가운데 정렬 */
  justify-content: center; /* 수평 방향으로 가운데 정렬 */
  height: 100%;
  text-align: center;
}

.bg-image {
  background-image: linear-gradient(
      0deg,
      rgba(217, 217, 217, 0.55),
      rgba(217, 217, 217, 0.55)
    ),
    url("~@/assets/coffeebeans.png");
  background-position: 50% 50%;
  background-size: cover;
  background-repeat: no-repeat;
}

.coffee-form {
  background: white;
  padding: 30px;
  border-radius: 8px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  width: 600px;
  margin-bottom: 20px; /* 폼과 버튼 사이의 간격 조정 */
}

.form-group {
  margin-bottom: 15px;
}

.form-group label {
  display: block;
  margin-bottom: 5px;
}

.form-group input {
  width: 100%;
  padding: 8px;
  box-sizing: border-box;
  border-radius: 4px;
  border: 1px solid #ccc;
}

.submit-btn {
  background-color: #60403b;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-weight: 700;
}

.submit-btn:hover {
  background-color: #50332f;
}

.coffee-knnet {
  margin-top: 20px; /* CoffeeKNNet 요소와 이전 요소들 사이의 간격 조정 */
}
</style>
