import nextra from 'nextra'

const withNextra = nextra({})
const basePath = process.env.BASE_PATH || ''

export default withNextra({
  output: 'export',
  images: {
    unoptimized: true,
  },
  basePath,
  trailingSlash: true,
})
