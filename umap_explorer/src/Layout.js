import React, { Component } from 'react'
import Sidebar from './Sidebar'
import Projection from './Projection'
import About from './About'
import * as _ from 'lodash'

// padding constructor
function p(tb, lr) {
  return `${tb}px ${lr}px`
}

let color_array = [[255, 255, 217], [254, 254, 215], [253, 254, 214], [253, 254, 213], [252, 254, 211], [252, 253, 210], [251, 253, 209], [251, 253, 208], [250, 253, 206], [249, 253, 205], [249, 252, 204], [248, 252, 203], [248, 252, 201], [247, 252, 200], [247, 251, 199], [246, 251, 198], [245, 251, 196], [245, 251, 195], [244, 251, 194], [244, 250, 193], [243, 250, 191], [243, 250, 190], [242, 250, 189], [242, 249, 188], [241, 249, 186], [240, 249, 185], [240, 249, 184], [239, 249, 183], [239, 248, 181], [238, 248, 180], [238, 248, 179], [237, 248, 178], [236, 247, 177], [235, 247, 177], [234, 247, 177], [233, 246, 177], [232, 246, 177], [230, 245, 177], [229, 245, 177], [228, 244, 177], [227, 244, 177], [226, 243, 177], [224, 243, 177], [223, 242, 178], [222, 242, 178], [221, 241, 178], [220, 241, 178], [218, 240, 178], [217, 240, 178], [216, 239, 178], [215, 239, 178], [214, 239, 178], [213, 238, 178], [211, 238, 178], [210, 237, 179], [209, 237, 179], [208, 236, 179], [207, 236, 179], [205, 235, 179], [204, 235, 179], [203, 234, 179], [202, 234, 179], [201, 233, 179], [199, 233, 179], [198, 232, 180], [196, 231, 180], [193, 231, 180], [191, 230, 180], [189, 229, 180], [187, 228, 181], [184, 227, 181], [182, 226, 181], [180, 225, 181], [178, 224, 182], [175, 223, 182], [173, 223, 182], [171, 222, 182], [169, 221, 182], [166, 220, 183], [164, 219, 183], [162, 218, 183], [160, 217, 183], [157, 216, 184], [155, 216, 184], [153, 215, 184], [151, 214, 184], [148, 213, 184], [146, 212, 185], [144, 211, 185], [141, 210, 185], [139, 209, 185], [137, 209, 185], [135, 208, 186], [132, 207, 186], [130, 206, 186], [128, 205, 186], [126, 204, 187], [124, 204, 187], [122, 203, 187], [120, 202, 187], [118, 201, 188], [116, 201, 188], [114, 200, 188], [112, 199, 189], [110, 198, 189], [108, 198, 189], [106, 197, 189], [104, 196, 190], [102, 196, 190], [100, 195, 190], [99, 194, 191], [97, 193, 191], [95, 193, 191], [93, 192, 191], [91, 191, 192], [89, 191, 192], [87, 190, 192], [85, 189, 193], [83, 188, 193], [81, 188, 193], [79, 187, 193], [77, 186, 194], [75, 185, 194], [73, 185, 194], [71, 184, 195], [69, 183, 195], [67, 183, 195], [65, 182, 195], [64, 181, 195], [63, 180, 195], [62, 179, 195], [61, 177, 195], [59, 176, 195], [58, 175, 195], [57, 174, 195], [56, 173, 195], [55, 172, 194], [54, 170, 194], [53, 169, 194], [52, 168, 194], [50, 167, 194], [49, 166, 194], [48, 165, 194], [47, 164, 194], [46, 162, 193], [45, 161, 193], [44, 160, 193], [42, 159, 193], [41, 158, 193], [40, 157, 193], [39, 155, 193], [38, 154, 193], [37, 153, 192], [36, 152, 192], [35, 151, 192], [33, 150, 192], [32, 148, 192], [31, 147, 192], [30, 146, 192], [29, 145, 192], [29, 144, 191], [29, 142, 190], [29, 140, 190], [29, 139, 189], [29, 137, 188], [29, 136, 187], [30, 134, 187], [30, 132, 186], [30, 131, 185], [30, 129, 184], [30, 128, 184], [30, 126, 183], [30, 124, 182], [31, 123, 181], [31, 121, 180], [31, 120, 180], [31, 118, 179], [31, 116, 178], [31, 115, 177], [32, 113, 177], [32, 112, 176], [32, 110, 175], [32, 108, 174], [32, 107, 174], [32, 105, 173], [33, 104, 172], [33, 102, 171], [33, 100, 171], [33, 99, 170], [33, 97, 169], [33, 96, 168], [33, 94, 168], [34, 93, 167], [34, 91, 166], [34, 90, 166], [34, 89, 165], [34, 87, 165], [34, 86, 164], [34, 85, 163], [34, 83, 163], [34, 82, 162], [34, 81, 161], [35, 79, 161], [35, 78, 160], [35, 77, 160], [35, 75, 159], [35, 74, 158], [35, 73, 158], [35, 71, 157], [35, 70, 156], [35, 69, 156], [35, 67, 155], [35, 66, 154], [36, 65, 154], [36, 64, 153], [36, 62, 153], [36, 61, 152], [36, 60, 151], [36, 58, 151], [36, 57, 150], [36, 56, 149], [36, 54, 149], [36, 53, 148], [36, 52, 148], [36, 51, 146], [35, 50, 144], [34, 49, 142], [33, 49, 140], [32, 48, 138], [31, 47, 136], [30, 47, 135], [29, 46, 133], [28, 45, 131], [28, 44, 129], [27, 44, 127], [26, 43, 125], [25, 42, 123], [24, 41, 121], [23, 41, 120], [22, 40, 118], [21, 39, 116], [20, 39, 114], [19, 38, 112], [18, 37, 110], [18, 36, 108], [17, 36, 106], [16, 35, 104], [15, 34, 103], [14, 34, 101], [13, 33, 99], [12, 32, 97], [11, 31, 95], [10, 31, 93], [9, 30, 91], [8, 29, 89], [8, 29, 88]]

class Layout extends Component {
  constructor(props) {
    super(props)
    this.state = {
      ww: null,
      wh: null,
      sidebar_height: null,
      hover_index: null,
      show_about: null,
      algorithm_choice: 0,
    }
    this.sidebar_ctx = null

    this.sidebar_ctx_an1 = null
    this.sidebar_ctx_an2 = null
    this.sidebar_ctx_an3 = null
    this.sidebar_ctx_an4 = null
    this.setSize = _.debounce(this.setSize.bind(this), 200)
    this.checkHash = this.checkHash.bind(this)
    this.setSidebarCanvas = this.setSidebarCanvas.bind(this)
    this.setSidebarCanvasAn1 = this.setSidebarCanvasAn1.bind(this)
    this.setSidebarCanvasAn2 = this.setSidebarCanvasAn2.bind(this)
    this.setSidebarCanvasAn3 = this.setSidebarCanvasAn3.bind(this)
    this.setSidebarCanvasAn4 = this.setSidebarCanvasAn4.bind(this)
    this.setSidebarCanvasAn5 = this.setSidebarCanvasAn5.bind(this)
    this.setSidebarCanvasAn6 = this.setSidebarCanvasAn6.bind(this)
    this.setSidebarCanvasAn7 = this.setSidebarCanvasAn7.bind(this)
    this.setSidebarCanvasAn8 = this.setSidebarCanvasAn8.bind(this)
    this.setSidebarCanvasAn9 = this.setSidebarCanvasAn9.bind(this)
    this.setSidebarCanvasAn10 = this.setSidebarCanvasAn10.bind(this)
    this.setSidebarCanvasAn11 = this.setSidebarCanvasAn11.bind(this)
    this.setSidebarCanvasAn12 = this.setSidebarCanvasAn12.bind(this)
    this.toggleAbout = this.toggleAbout.bind(this)
    this.selectAlgorithm = this.selectAlgorithm.bind(this)
  }

  selectAlgorithm(v) {
    let i = this.props.algorithm_options.indexOf(v)
    this.setState({ algorithm_choice: i })
  }

  setSize() {
    this.setState({ ww: window.innerWidth, wh: window.innerHeight })
    let sidebar_height = this.sidebar_mount.offsetHeight
    this.setState({ sidebar_height: sidebar_height })
    if (this.sidebar_ctx) this.sidebar_ctx.imageSmoothingEnabled = false
  }

  setSidebarCanvas(canvas) {
    let ctx = canvas.getContext('2d')
    ctx.imageSmoothingEnabled = false
    this.sidebar_ctx = ctx
  }

  setSidebarCanvasAn1(canvas) {
    let ctx = canvas.getContext('2d')
    ctx.imageSmoothingEnabled = false
    this.sidebar_ctx_an1 = ctx
  }

  setSidebarCanvasAn2(canvas) {
    let ctx = canvas.getContext('2d')
    ctx.imageSmoothingEnabled = false
    this.sidebar_ctx_an2 = ctx
  }

  setSidebarCanvasAn3(canvas) {
    let ctx = canvas.getContext('2d')
    ctx.imageSmoothingEnabled = false
    this.sidebar_ctx_an3 = ctx
  }

  setSidebarCanvasAn4(canvas) {
    let ctx = canvas.getContext('2d')
    ctx.imageSmoothingEnabled = false
    this.sidebar_ctx_an4 = ctx
  }

  setSidebarCanvasAn5(canvas) {
    let ctx = canvas.getContext('2d')
    ctx.imageSmoothingEnabled = false
    this.sidebar_ctx_an5 = ctx
  }

  setSidebarCanvasAn6(canvas) {
    let ctx = canvas.getContext('2d')
    ctx.imageSmoothingEnabled = false
    this.sidebar_ctx_an6 = ctx
  }

  setSidebarCanvasAn7(canvas) {
    let ctx = canvas.getContext('2d')
    ctx.imageSmoothingEnabled = false
    this.sidebar_ctx_an7 = ctx
  }

  setSidebarCanvasAn8(canvas) {
    let ctx = canvas.getContext('2d')
    ctx.imageSmoothingEnabled = false
    this.sidebar_ctx_an8 = ctx
  }

  setSidebarCanvasAn9(canvas) {
    let ctx = canvas.getContext('2d')
    ctx.imageSmoothingEnabled = false
    this.sidebar_ctx_an9 = ctx
  }

  setSidebarCanvasAn10(canvas) {
    let ctx = canvas.getContext('2d')
    ctx.imageSmoothingEnabled = false
    this.sidebar_ctx_an10 = ctx
  }

  setSidebarCanvasAn11(canvas) {
    let ctx = canvas.getContext('2d')
    ctx.imageSmoothingEnabled = false
    this.sidebar_ctx_an11 = ctx
  }

  setSidebarCanvasAn12(canvas) {
    let ctx = canvas.getContext('2d')
    ctx.imageSmoothingEnabled = false
    this.sidebar_ctx_an12 = ctx
  }

  toggleAbout(state) {
    if (state === true) {
      window.history.pushState(null, 'About UMAP Explorer', '#about')
      this.setState({ show_about: true })
    } else if (state === false) {
      window.history.pushState(null, 'UMAP Explorer', window.location.pathname)
      this.setState({ show_about: false })
    }
  }

  setHoverIndex(hover_index) {
    this.setState({ hover_index: hover_index })
  }

  componentWillMount() {
    this.setSize()
    this.checkHash()
  }

  checkHash() {
    if (window.location.hash && window.location.hash === '#about') {
      this.setState({ show_about: true })
    } else {
      this.setState({ show_about: false })
    }
  }

  componentDidMount() {
    window.addEventListener('resize', this.setSize)
    window.addEventListener('popstate', this.checkHash)
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.setSize)
  }

  render() {
    let {
      umap_dims_0_2,
      umap_dims_1_2,
      umap_dims_0_1,
      mnist_labels,
      mnist_dates,
      algorithm_options,
      algorithm_embedding_keys,
    } = this.props
    let {
      ww,
      wh,
      sidebar_height,
      hover_index,
      show_about,
      algorithm_choice,
    } = this.state
    let sidebar_ctx = this.sidebar_ctx
    let sidebar_ctx_an1 = this.sidebar_ctx_an1
    let sidebar_ctx_an2 = this.sidebar_ctx_an2
    let sidebar_ctx_an3 = this.sidebar_ctx_an3
    let sidebar_ctx_an4 = this.sidebar_ctx_an4
    let sidebar_ctx_an5 = this.sidebar_ctx_an5
    let sidebar_ctx_an6 = this.sidebar_ctx_an6
    let sidebar_ctx_an7 = this.sidebar_ctx_an7
    let sidebar_ctx_an8 = this.sidebar_ctx_an8
    let sidebar_ctx_an9 = this.sidebar_ctx_an9
    let sidebar_ctx_an10 = this.sidebar_ctx_an10
    let sidebar_ctx_an11 = this.sidebar_ctx_an11
    let sidebar_ctx_an12 = this.sidebar_ctx_an12

    let line_height = 1.5

    let sidebar_style = {
      position: 'absolute',
      left: 0,
      top: 0,
      height: '100vh',
      overflow: 'auto',
      background: '#222',
      display: 'flex',
      flexDirection: 'column',
    }
    let main_style = {
      position: 'relative',
      height: '100vh',
      background: '#111',
      overflow: 'hidden',
    }

    let sidebar_image_size, sidebar_orientation
    let font_size = 16

    //if (ww < 800) {
    font_size = 14
    sidebar_style = {
      ...sidebar_style,
      flexDirection: 'row',
      width: '100%',
      top: 'auto',
      height: 'auto',
      bottom: 0,
    }
    main_style = { width: ww, height: wh - sidebar_height }
    sidebar_image_size = font_size * line_height * 6
    sidebar_orientation = 'horizontal'
    // } else if (ww < 800 + 600) {
    //   let scaler = 200 + (300 - 200) * ((ww - 800) / 600)
    //   font_size = 14 + 2 * ((ww - 800) / 600)
    //   sidebar_style = {
    //     ...sidebar_style,
    //     width: scaler,
    //   }
    //   sidebar_image_size = sidebar_style.width
    //   main_style = {
    //     ...main_style,
    //     width: ww - scaler,
    //     left: scaler,
    //     height: wh,
    //   }
    //   sidebar_orientation = 'vertical'
    // } else {
    //   sidebar_style = {
    //     ...sidebar_style,
    //     width: 300,
    //   }
    //   main_style = {
    //     ...main_style,
    //     width: ww - 300,
    //     left: 300,
    //     height: wh,
    //   }
    //   sidebar_image_size = sidebar_style.width
    //   sidebar_orientation = 'vertical'
    // }

    let grem = font_size * line_height

    let general_style = {
      fontSize: font_size,
      lineHeight: line_height,
    }

    return ww !== null ? (
      <div style={general_style}>
        <div
          style={sidebar_style}
          ref={sidebar_mount => {
            this.sidebar_mount = sidebar_mount
          }}
        >
          <Sidebar
            sidebar_orientation={sidebar_orientation}
            sidebar_image_size={sidebar_image_size}
            grem={grem}
            p={p}
            color_array={color_array}
            setSidebarCanvas={this.setSidebarCanvas}
            setSidebarCanvasAn1={this.setSidebarCanvasAn1}
            setSidebarCanvasAn2={this.setSidebarCanvasAn2}
            setSidebarCanvasAn3={this.setSidebarCanvasAn3}
            setSidebarCanvasAn4={this.setSidebarCanvasAn4}
            setSidebarCanvasAn5={this.setSidebarCanvasAn5}
            setSidebarCanvasAn6={this.setSidebarCanvasAn6}
            setSidebarCanvasAn7={this.setSidebarCanvasAn7}
            setSidebarCanvasAn8={this.setSidebarCanvasAn8}
            setSidebarCanvasAn9={this.setSidebarCanvasAn9}
            setSidebarCanvasAn10={this.setSidebarCanvasAn10}
            setSidebarCanvasAn11={this.setSidebarCanvasAn11}
            setSidebarCanvasAn12={this.setSidebarCanvasAn12}
            hover_index={hover_index}
            mnist_labels={mnist_labels}
            mnist_dates={mnist_dates}
            toggleAbout={this.toggleAbout}
            algorithm_options={algorithm_options}
            algorithm_choice={algorithm_choice}
            selectAlgorithm={this.selectAlgorithm}
          />
        </div>
        <div style={main_style}>
          <Projection
            width={main_style.width}
            height={main_style.height}
            umap_dims_0_2={umap_dims_0_2}
            umap_dims_1_2={umap_dims_1_2}
            umap_dims_0_1={umap_dims_0_1}
            mnist_labels={mnist_labels}
            color_array={color_array}
            sidebar_ctx={sidebar_ctx}
            sidebar_ctx_an1={sidebar_ctx_an1}
            sidebar_ctx_an2={sidebar_ctx_an2}
            sidebar_ctx_an3={sidebar_ctx_an3}
            sidebar_ctx_an4={sidebar_ctx_an4}
            sidebar_ctx_an5={sidebar_ctx_an5}
            sidebar_ctx_an6={sidebar_ctx_an6}
            sidebar_ctx_an7={sidebar_ctx_an7}
            sidebar_ctx_an8={sidebar_ctx_an8}
            sidebar_ctx_an9={sidebar_ctx_an9}
            sidebar_ctx_an10={sidebar_ctx_an10}
            sidebar_ctx_an11={sidebar_ctx_an11}
            sidebar_ctx_an12={sidebar_ctx_an12}
            sidebar_image_size={sidebar_image_size}
            setHoverIndex={this.setHoverIndex.bind(this)}
            algorithm_embedding_keys={algorithm_embedding_keys}
            algorithm_choice={algorithm_choice}
          />
        </div>
        {show_about ? (
          <About grem={grem} p={p} toggleAbout={this.toggleAbout} />
        ) : null}
      </div>
    ) : (
        <div style={{ padding: '1rem' }}>Loading layout...</div>
      )
  }
}

export default Layout
