import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const routes = [
  {
    path: '/',
    name: 'main',
    component: () => import('../views/MainView.vue')
  },
  // CoffeeForm
  {
    path: '/CoffeeFormView',
    name: 'CoffeeFormView',
    component: () => import('../views/CoffeeFormView.vue')
  },
  // CoffeeResult
  {
    path: '/CoffeeResult',
    name: 'ResultView',
    component: () => import('../views/ResultView.vue')
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
