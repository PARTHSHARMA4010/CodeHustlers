/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
        extend: {
          colors: {
            primary: '#006953',
    secondary: '#A9C7B8',
    background: '#F7F6F2'
          },
        },
      },
  plugins: [],
}


// module.exports = {
//   content: [
//     "./index.html",
//     "./src/*/.{js,ts,jsx,tsx}",
//   ],
//   theme: {
//     extend: {
//       colors: {
//         primary: '#006953',
// secondary: '#A9C7B8',
// background: '#F7F6F2'
//       },
//     },
//   },
//   plugins: [],
// }